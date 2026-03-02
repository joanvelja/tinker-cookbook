"""Wave 3 gate: adversarial parity tests for Inspect eval integration.

Validates:
- Lint cleanliness (run externally)
- GPQAAdapter free_debate mode emits empty answer strings
- Metric extraction bug fix (metric_name used as key, not score_result.name)
- open_reasoning parameter threading through solver/eval/spec
- randomize_position parameter threading and store persistence
- Identity-aware scoring (id/ prefixed keys, remap correctness)
- Default values match training expectations
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.env import (
    IDENTITY_REMAP_BASES,
    _remap_to_identity,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.eval.dataset_adapter import GPQAAdapter
from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
    DebateInspectEvaluatorBuilder,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import (
    _identity_metric_keys,
    _state_to_json,
    _STORE_KEY,
    _TRAINED_ROLE_KEY,
    debate_eval,
    debate_scorer,
    debate_solver,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.metrics import (
    mcq_debate_metrics,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome,
    DebateSpec,
    DebateState,
    JudgeDecision,
    Phase,
    ProtocolKind,
    Role,
    Utterance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    target: str | None = "A",
    num_rounds: int = 2,
    open_reasoning: bool = True,
    prompts_ref: str = "scientific_mcq",
) -> DebateSpec:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, num_rounds)
    return DebateSpec(
        debate_id="parity-test-001",
        task_prompt="What is X?\nA) Foo\nB) Bar\nC) Baz\nD) Qux",
        answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
        schedule=schedule,
        open_reasoning=open_reasoning,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref=prompts_ref,
        target=target,
    )


def _make_utterance(
    role: Role,
    round_index: int,
    phase: Phase,
    text: str,
    *,
    answer: str | None = None,
    slot_id: int = 0,
) -> Utterance:
    fields = {"answer": answer} if answer is not None else None
    return Utterance(
        role=role,
        round_index=round_index,
        phase=phase,
        text=text,
        token_count=len(text.split()),
        slot_id=slot_id,
        fields=fields,
    )


def _make_completed_state(
    *,
    target: str | None = "A",
    a_answer: str = "A",
    b_answer: str = "B",
    winner: Role | None = Role.DEBATER_A,
) -> DebateState:
    spec = _make_spec(target=target, num_rounds=2)
    transcript = (
        _make_utterance(
            Role.DEBATER_A, 0, Phase.PROPOSE, "I argue for A", answer=a_answer, slot_id=0
        ),
        _make_utterance(
            Role.DEBATER_B, 0, Phase.PROPOSE, "I argue for B", answer=b_answer, slot_id=1
        ),
        _make_utterance(
            Role.DEBATER_A, 1, Phase.CRITIQUE, "Rebutting B", answer=a_answer, slot_id=2
        ),
        _make_utterance(
            Role.DEBATER_B, 1, Phase.CRITIQUE, "Rebutting A", answer=b_answer, slot_id=3
        ),
    )
    judge_trace = (
        JudgeDecision(
            round_index=0,
            verdict="debater_a wins",
            score_delta_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
        ),
    )
    outcome = DebateOutcome(
        winner=winner,
        scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
        verdict_text="Debater A wins",
    )
    return DebateState(
        spec=spec,
        slot_index=4,
        rounds_completed=2,
        transcript=transcript,
        pending_simultaneous={},
        judge_trace=judge_trace,
        done=True,
        outcome=outcome,
    )


# ===================================================================
# 1. GPQAAdapter free_debate mode
# ===================================================================


class TestGPQAAdapterFreeDebate:
    def test_free_debate_attribute_exists(self):
        """GPQAAdapter accepts free_debate parameter."""
        adapter = GPQAAdapter(free_debate=True)
        assert adapter._free_debate is True

    def test_free_debate_default_is_false(self):
        adapter = GPQAAdapter()
        assert adapter._free_debate is False

    def test_free_debate_emits_empty_answers(self):
        """free_debate=True produces empty answer_a/answer_b metadata."""
        try:
            adapter = GPQAAdapter(free_debate=True, limit=1, seed=0)
            samples = adapter.to_samples()
        except Exception:
            pytest.skip("GPQA dataset not available (network required)")

        assert len(samples) == 1
        s = samples[0]
        assert s.metadata["answer_a"] == ""
        assert s.metadata["answer_b"] == ""
        assert s.metadata["source"] == "gpqa_diamond"
        # Target should still be set (correct answer label)
        assert s.target in ("A", "B", "C", "D")

    def test_normal_mode_has_nonempty_answers(self):
        """Normal mode (free_debate=False) has non-empty answer assignments."""
        try:
            adapter = GPQAAdapter(free_debate=False, limit=1, seed=0)
            samples = adapter.to_samples()
        except Exception:
            pytest.skip("GPQA dataset not available (network required)")

        s = samples[0]
        assert s.metadata["answer_a"] != ""
        assert s.metadata["answer_b"] != ""
        assert s.metadata["answer_a"] != s.metadata["answer_b"]


# ===================================================================
# 1b. _drive_turn structured content handling
# ===================================================================


class TestDriveTurnStructuredContent:
    """Regression test: _drive_turn must handle list[ContentPart] responses."""

    @pytest.mark.asyncio
    async def test_drive_turn_handles_structured_content(self):
        """Completer returning list[ContentPart] (thinking model) should not crash.

        Before the fix, `response["content"]` was a list passed directly to
        `runtime.submit()` which calls `strip_think(text)` expecting str.
        """
        from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
        from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import _drive_turn

        spec = _make_spec(num_rounds=1)
        initial_state = DebateState(
            spec=spec,
            slot_index=0,
            rounds_completed=0,
            transcript=(),
            pending_simultaneous={},
            judge_trace=(),
            done=False,
            outcome=None,
        )

        runtime = DebateRuntime(initial_state, judge_callback=AsyncMock())

        # Simulate a thinking-model response: content is list[ContentPart]
        structured_content = [
            {"type": "thinking", "thinking": "Let me reason about this..."},
            {"type": "text", "text": "I argue for answer A."},
        ]
        mock_completer = AsyncMock(
            return_value={"role": "assistant", "content": structured_content}
        )

        with patch("tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task.span") as mock_span:
            mock_span.return_value.__aenter__ = AsyncMock()
            mock_span.return_value.__aexit__ = AsyncMock()
            with patch(
                "tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task.transcript"
            ) as mock_transcript:
                mock_transcript.return_value.info = MagicMock()
                result = await _drive_turn(runtime, Role.DEBATER_A, mock_completer)

        assert result is True
        # The text stored in the utterance should be a string, not a list
        assert len(runtime.state.transcript) == 1
        utt = runtime.state.transcript[0]
        assert isinstance(utt.text, str)
        assert "<think>" in utt.text
        assert "I argue for answer A." in utt.text

    @pytest.mark.asyncio
    async def test_drive_turn_handles_plain_string_content(self):
        """Plain string content (non-thinking model) should still work."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
        from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import _drive_turn

        spec = _make_spec(num_rounds=1)
        initial_state = DebateState(
            spec=spec,
            slot_index=0,
            rounds_completed=0,
            transcript=(),
            pending_simultaneous={},
            judge_trace=(),
            done=False,
            outcome=None,
        )

        runtime = DebateRuntime(initial_state, judge_callback=AsyncMock())

        mock_completer = AsyncMock(
            return_value={"role": "assistant", "content": "I argue for answer A."}
        )

        with patch("tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task.span") as mock_span:
            mock_span.return_value.__aenter__ = AsyncMock()
            mock_span.return_value.__aexit__ = AsyncMock()
            with patch(
                "tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task.transcript"
            ) as mock_transcript:
                mock_transcript.return_value.info = MagicMock()
                result = await _drive_turn(runtime, Role.DEBATER_A, mock_completer)

        assert result is True
        utt = runtime.state.transcript[0]
        assert isinstance(utt.text, str)
        assert utt.text == "I argue for answer A."


# ===================================================================
# 2. Metric extraction bug fix
# ===================================================================


class TestMetricExtraction:
    """Verify evaluator metric extraction uses metric_name, not score_result.name."""

    def test_metric_extraction_source_code(self):
        """Static check: the metric extraction loop uses metric_name as dict key."""
        import tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator as mod

        source = inspect.getsource(mod.DebateInspectEvaluator.__call__)
        tree = ast.parse(textwrap.dedent(source))

        # Find: for metric_name, metric in score_result.metrics.items():
        #            metrics[metric_name] = metric.value
        # The bug was: metrics[score_result.name] = metric.value
        found_correct = False
        found_bug = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        # Check what the subscript key is
                        if isinstance(target.slice, ast.Name):
                            if target.slice.id == "metric_name":
                                found_correct = True
                            # The bug pattern: score_result.name as key
                        if isinstance(target.slice, ast.Attribute):
                            if (
                                isinstance(target.slice.value, ast.Name)
                                and target.slice.value.id == "score_result"
                                and target.slice.attr == "name"
                            ):
                                found_bug = True

        assert found_correct, "metric_name should be used as dict key in metric extraction loop"
        assert not found_bug, (
            "score_result.name must NOT be used as dict key (last-metric-wins bug)"
        )

    @pytest.mark.asyncio
    async def test_metric_extraction_mock_3_metrics(self):
        """Mock score_result with 3 metrics; verify all 3 appear in output."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
            DebateInspectEvaluatorBuilder,
        )

        # Build mock eval result with 3 metrics under one score_result
        mock_metric_a = MagicMock()
        mock_metric_a.value = 0.75
        mock_metric_b = MagicMock()
        mock_metric_b.value = 0.5
        mock_metric_c = MagicMock()
        mock_metric_c.value = 0.9

        mock_score_result = MagicMock()
        mock_score_result.name = "debate_scorer"  # This was the bug key
        mock_score_result.metrics = {
            "accuracy.debater_a": mock_metric_a,
            "judge_quality": mock_metric_b,
            "truth_surfaced": mock_metric_c,
        }

        mock_task_result = MagicMock()
        mock_task_result.results = MagicMock()
        mock_task_result.results.scores = [mock_score_result]

        adapter = MagicMock()
        adapter.to_samples.return_value = [
            MagicMock(
                input="q", target="A", metadata={"answer_a": "A", "answer_b": "B", "source": "test"}
            )
        ]

        builder = DebateInspectEvaluatorBuilder(
            adapter=adapter,
            renderer_name="qwen3",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
        )
        evaluator = builder()

        _EVAL_MODULE = "tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator"
        with (
            patch(
                f"{_EVAL_MODULE}.eval_async",
                new_callable=AsyncMock,
                return_value=[mock_task_result],
            ),
            patch(f"{_EVAL_MODULE}.get_renderer", return_value=MagicMock()),
            patch(f"{_EVAL_MODULE}.get_tokenizer", return_value=MagicMock()),
            patch(f"{_EVAL_MODULE}.TinkerMessageCompleter", return_value=MagicMock()),
            patch(f"{_EVAL_MODULE}.tinker.ServiceClient", return_value=MagicMock()),
        ):
            metrics = await evaluator(MagicMock())

        # All 3 metrics must appear with correct keys (not collapsed to "debate_scorer")
        assert len(metrics) == 3, f"Expected 3 metrics, got {len(metrics)}: {metrics}"
        assert metrics["accuracy.debater_a"] == 0.75
        assert metrics["judge_quality"] == 0.5
        assert metrics["truth_surfaced"] == 0.9
        # The bug would produce: {"debate_scorer": 0.9} (only last metric)
        assert "debate_scorer" not in metrics


# ===================================================================
# 3. open_reasoning parameter threading
# ===================================================================


class TestOpenReasoningThreading:
    def test_debate_solver_accepts_open_reasoning(self):
        """debate_solver has open_reasoning in its signature."""
        sig = inspect.signature(debate_solver)
        assert "open_reasoning" in sig.parameters

    def test_debate_eval_accepts_open_reasoning(self):
        """debate_eval has open_reasoning in its signature."""
        sig = inspect.signature(debate_eval)
        assert "open_reasoning" in sig.parameters

    def test_evaluator_builder_has_open_reasoning(self):
        """DebateInspectEvaluatorBuilder has open_reasoning field."""
        import typing

        assert hasattr(DebateInspectEvaluatorBuilder, "open_reasoning")
        hints = typing.get_type_hints(DebateInspectEvaluatorBuilder)
        assert "open_reasoning" in hints

    def test_solver_threads_open_reasoning_to_spec(self):
        """Static check: solver builds DebateSpec with open_reasoning param, not hardcoded."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.eval import inspect_task as mod

        source = inspect.getsource(mod.debate_solver)
        # The spec construction should use the open_reasoning parameter
        assert "open_reasoning=open_reasoning" in source, (
            "DebateSpec should use the open_reasoning parameter, not a hardcoded value"
        )
        # It should NOT be hardcoded True
        assert (
            "open_reasoning=True" not in source.split("DebateSpec(")[1].split(")")[0]
            if "DebateSpec(" in source
            else True
        )

    def test_evaluator_threads_open_reasoning(self):
        """Static check: evaluator passes open_reasoning from config to debate_eval."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.eval import evaluator as mod

        source = inspect.getsource(mod.DebateInspectEvaluator.__call__)
        assert (
            "open_reasoning=cfg.open_reasoning" in source
            or "open_reasoning=self._config.open_reasoning" in source
        )


# ===================================================================
# 4. randomize_position parameter threading
# ===================================================================


class TestRandomizePositionThreading:
    def test_debate_solver_accepts_randomize_position(self):
        sig = inspect.signature(debate_solver)
        assert "randomize_position" in sig.parameters

    def test_debate_eval_accepts_randomize_position(self):
        sig = inspect.signature(debate_eval)
        assert "randomize_position" in sig.parameters

    def test_evaluator_builder_has_randomize_position(self):
        import typing

        hints = typing.get_type_hints(DebateInspectEvaluatorBuilder)
        assert "randomize_position" in hints

    def test_solver_stores_trained_role(self):
        """Static check: solver stores trained_role in TaskState store."""
        import tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task as mod

        source = inspect.getsource(mod)
        assert _TRAINED_ROLE_KEY in source
        assert "state.store.set" in source
        assert "trained_role" in source

    def test_solver_randomizes_when_flag_true(self):
        """Static check: when randomize_position, trained_role is randomly chosen."""
        import tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task as mod

        source = inspect.getsource(mod)
        assert "random.choice" in source
        assert "randomize_position" in source


# ===================================================================
# 5. Identity-aware scoring
# ===================================================================


class TestIdentityAwareScoring:
    def test_scorer_reads_trained_role_from_store(self):
        """Static check: scorer reads _TRAINED_ROLE_KEY from store."""
        import tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task as mod

        source = inspect.getsource(mod)
        assert _TRAINED_ROLE_KEY in source
        assert "_remap_to_identity" in source

    def test_identity_metric_keys_format(self):
        """id/ prefixed keys are generated for all IDENTITY_REMAP_BASES."""
        keys = _identity_metric_keys()
        assert "id/trained_role_is_a" in keys
        for base in IDENTITY_REMAP_BASES:
            assert f"id/{base}.trained" in keys
            assert f"id/{base}.opponent" in keys

    def test_identity_metric_keys_count(self):
        """Expected count: 1 (trained_role_is_a) + 2 per base."""
        keys = _identity_metric_keys()
        expected = 1 + 2 * len(IDENTITY_REMAP_BASES)
        assert len(keys) == expected

    @pytest.mark.asyncio
    async def test_scorer_produces_identity_keys(self):
        """Scorer output contains id/ prefixed keys when trained_role is set."""
        from inspect_ai.scorer import Target
        from inspect_ai.solver import TaskState

        state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
        json_str = _state_to_json(state)

        task_state = MagicMock(spec=TaskState)
        store_data = {
            _STORE_KEY: json_str,
            _TRAINED_ROLE_KEY: Role.DEBATER_A.value,
        }
        task_state.store = MagicMock()
        task_state.store.get = MagicMock(
            side_effect=lambda k, default=None: store_data.get(k, default)
        )

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        assert isinstance(score.value, dict)
        # Must contain id/ keys
        id_keys = [k for k in score.value if k.startswith("id/")]
        assert len(id_keys) > 0, (
            f"Expected id/ keys in scorer output, got: {list(score.value.keys())}"
        )
        assert "id/trained_role_is_a" in score.value
        assert score.value["id/trained_role_is_a"] == 1.0  # trained=A

    @pytest.mark.asyncio
    async def test_identity_remap_swaps_when_trained_is_b(self):
        """When trained_role=B, id/accuracy.trained should map to debater_b."""
        from inspect_ai.scorer import Target
        from inspect_ai.solver import TaskState

        state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
        json_str = _state_to_json(state)

        task_state = MagicMock(spec=TaskState)
        store_data = {
            _STORE_KEY: json_str,
            _TRAINED_ROLE_KEY: Role.DEBATER_B.value,
        }
        task_state.store = MagicMock()
        task_state.store.get = MagicMock(
            side_effect=lambda k, default=None: store_data.get(k, default)
        )

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        v = score.value
        assert v["id/trained_role_is_a"] == 0.0  # trained=B
        # accuracy.debater_a=1.0 (correct), accuracy.debater_b=0.0 (wrong)
        # With trained=B: id/accuracy.trained = debater_b = 0.0
        assert v["id/accuracy.trained"] == 0.0
        assert v["id/accuracy.opponent"] == 1.0

    @pytest.mark.asyncio
    async def test_identity_remap_preserves_original_keys(self):
        """Identity remap adds id/ keys but preserves original seat-based keys."""
        from inspect_ai.scorer import Target
        from inspect_ai.solver import TaskState

        state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
        json_str = _state_to_json(state)

        task_state = MagicMock(spec=TaskState)
        store_data = {
            _STORE_KEY: json_str,
            _TRAINED_ROLE_KEY: Role.DEBATER_A.value,
        }
        task_state.store = MagicMock()
        task_state.store.get = MagicMock(
            side_effect=lambda k, default=None: store_data.get(k, default)
        )

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        v = score.value
        # Original seat-based keys should still be present
        assert "accuracy.debater_a" in v
        assert "accuracy.debater_b" in v
        assert "win_rate.debater_a" in v


# ===================================================================
# 6. Default values match training
# ===================================================================


class TestDefaultValues:
    """Verify defaults by instantiating with only required args (adapter)."""

    @pytest.fixture()
    def builder(self):
        adapter = MagicMock()
        return DebateInspectEvaluatorBuilder(adapter=adapter)

    def test_prompts_ref_default(self, builder):
        """Default prompts_ref should be 'judge_exploit', not 'scientific_mcq'."""
        assert builder.prompts_ref == "judge_exploit"

    def test_opponent_max_tokens_default(self, builder):
        """Default opponent_max_tokens should be 8192."""
        assert builder.opponent_max_tokens == 8192

    def test_judge_max_tokens_default(self, builder):
        """Default judge_max_tokens should be 4096."""
        assert builder.judge_max_tokens == 4096

    def test_open_reasoning_default_false(self, builder):
        """Default open_reasoning should be False."""
        assert builder.open_reasoning is False

    def test_randomize_position_default_true(self, builder):
        """Default randomize_position should be True."""
        assert builder.randomize_position is True

    def test_num_rounds_default(self, builder):
        assert builder.num_rounds == 2

    def test_protocol_kind_default(self, builder):
        assert builder.protocol_kind == ProtocolKind.SEQUENTIAL


# ===================================================================
# 7. Scorer metric aggregation registration
# ===================================================================


class TestScorerMetricAggregation:
    def test_scorer_registers_all_mcq_metric_keys(self):
        """The scorer @scorer decorator should register aggregation for all mcq metrics."""
        # The scorer calls mcq_debate_metrics() for metric names.
        # Verify the identity keys cover all IDENTITY_REMAP_BASES.
        default_metrics = mcq_debate_metrics()
        _identity_metric_keys()  # ensure it doesn't error

        # Every base in IDENTITY_REMAP_BASES should correspond to a metric in defaults
        for base in IDENTITY_REMAP_BASES:
            a_key = f"{base}.debater_a"
            b_key = f"{base}.debater_b"
            assert a_key in default_metrics or b_key in default_metrics, (
                f"IDENTITY_REMAP_BASES entry '{base}' has no corresponding metric"
            )

    def test_remap_identity_unit(self):
        """Direct test of _remap_to_identity function."""
        m = {
            "accuracy.debater_a": 0.8,
            "accuracy.debater_b": 0.3,
            "win_rate.debater_a": 1.0,
            "win_rate.debater_b": 0.0,
        }
        result = _remap_to_identity(m, Role.DEBATER_A)
        assert result["id/trained_role_is_a"] == 1.0
        assert result["id/accuracy.trained"] == 0.8
        assert result["id/accuracy.opponent"] == 0.3
        assert result["id/win_rate.trained"] == 1.0
        assert result["id/win_rate.opponent"] == 0.0
        # Originals preserved
        assert result["accuracy.debater_a"] == 0.8

        # Flip: trained=B
        result_b = _remap_to_identity(m, Role.DEBATER_B)
        assert result_b["id/trained_role_is_a"] == 0.0
        assert result_b["id/accuracy.trained"] == 0.3
        assert result_b["id/accuracy.opponent"] == 0.8
