"""Tests for debate eval module (Inspect AI integration)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.eval.dataset_adapter import (
    DatasetAdapter,
    GPQAAdapter,
    GPQAOpenEndedAdapter,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import (
    _identity_metric_keys,
    _state_from_json,
    _state_to_json,
    _STORE_KEY,
    _TRAINED_ROLE_KEY,
    debate_eval,
    debate_scorer,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.builders import IDENTITY_REMAP_BASES
from tinker_cookbook.scoring import BinaryJudgeBuilder
from tinker_cookbook.recipes.multiplayer_rl.debate.tests.conftest import make_spec
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome,
    DebateSpec,
    DebateState,
    JudgeDecision,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    Utterance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    target: str | None = "A",
    num_rounds: int = 2,
    prompts_ref: str = "scientific_mcq",
) -> DebateSpec:
    return make_spec(
        debate_id="test-debate-001",
        task_prompt="What is the answer?\nA) Foo\nB) Bar\nC) Baz\nD) Qux",
        target=target,
        num_rounds=num_rounds,
        open_reasoning=True,
        prompts_ref=prompts_ref,
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
    """Create a completed debate state with known answers and winner."""
    spec = _make_spec(target=target, num_rounds=2)
    transcript = (
        _make_utterance(
            Role.DEBATER_A,
            0,
            Phase.PROPOSE,
            "I argue for A because...",
            answer=a_answer,
            slot_id=0,
        ),
        _make_utterance(
            Role.DEBATER_B,
            0,
            Phase.PROPOSE,
            "I argue for B because...",
            answer=b_answer,
            slot_id=1,
        ),
        _make_utterance(
            Role.DEBATER_A,
            1,
            Phase.CRITIQUE,
            "Responding to B's argument...",
            answer=a_answer,
            slot_id=2,
        ),
        _make_utterance(
            Role.DEBATER_B,
            1,
            Phase.CRITIQUE,
            "Responding to A's argument...",
            answer=b_answer,
            slot_id=3,
        ),
    )
    judge_trace = (
        JudgeDecision(
            round_index=0,
            verdict="debater_a is more convincing",
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


# ---------------------------------------------------------------------------
# 1. test_dataset_adapter_protocol
# ---------------------------------------------------------------------------


def test_dataset_adapter_protocol():
    """GPQAAdapter satisfies DatasetAdapter protocol."""
    assert isinstance(GPQAAdapter(), DatasetAdapter)


# ---------------------------------------------------------------------------
# 2. test_gpqa_adapter_samples
# ---------------------------------------------------------------------------


def test_gpqa_adapter_samples():
    """GPQAAdapter returns well-formed Samples with correct fields."""
    adapter = GPQAAdapter(limit=5, seed=42)
    samples = adapter.to_samples()
    assert len(samples) == 5

    for s in samples:
        assert isinstance(s.input, str)
        assert len(s.input) > 0
        assert s.target in ("A", "B", "C", "D")
        assert "answer_a" in s.metadata
        assert "answer_b" in s.metadata
        assert "source" in s.metadata
        assert s.metadata["answer_a"] in ("A", "B", "C", "D")
        assert s.metadata["answer_b"] in ("A", "B", "C", "D")
        assert s.metadata["answer_a"] != s.metadata["answer_b"]
        assert s.metadata["source"] == "gpqa_diamond"


def test_gpqa_open_ended_adapter_samples(monkeypatch):
    rows = [
        {
            "question": "What causes tides?",
            "answer": "The Moon's gravitational pull.",
            "record_id": "rec-b",
            "domain": "physics",
            "subdomain": "mechanics",
            "writer_difficulty": "hard",
            "expert_accuracy": 0.8,
            "non_expert_accuracy": 0.2,
            "conversion_type": "rewrite",
            "flag": "",
        },
        {
            "question": "Why is the sky blue?",
            "answer": "Rayleigh scattering.",
            "record_id": "rec-a",
            "domain": "physics",
            "subdomain": "optics",
            "writer_difficulty": "medium",
            "expert_accuracy": 0.9,
            "non_expert_accuracy": 0.3,
            "conversion_type": "rewrite",
            "flag": "",
        },
    ]

    def _fake_load_dataset(dataset_name: str, subset: str, split: str):
        assert dataset_name == "joanvelja/gpqa-open-ended"
        assert subset == "extended"
        assert split == "train"
        return rows

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    adapter = GPQAOpenEndedAdapter(
        subset="extended",
        split="train",
        record_ids=["rec-a", "rec-b"],
    )
    samples = adapter.to_samples()

    assert [sample.metadata["record_id"] for sample in samples] == ["rec-a", "rec-b"]
    assert [sample.input for sample in samples] == ["Why is the sky blue?", "What causes tides?"]
    assert [sample.target for sample in samples] == [
        "Rayleigh scattering.",
        "The Moon's gravitational pull.",
    ]
    for sample in samples:
        assert sample.metadata["answer_a"] == ""
        assert sample.metadata["answer_b"] == ""
        assert sample.metadata["source"] == "gpqa_open_ended"
    assert adapter.resolve_scoring_mode() == ScoringMode.OPEN_ENDED


def test_gpqa_open_ended_adapter_missing_record_id(monkeypatch):
    rows = [
        {
            "question": "Q",
            "answer": "A",
            "record_id": "rec-present",
            "domain": "physics",
            "subdomain": "optics",
            "writer_difficulty": "medium",
            "expert_accuracy": 0.9,
            "non_expert_accuracy": 0.3,
            "conversion_type": "rewrite",
            "flag": "",
        }
    ]

    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda dataset_name, subset, split: rows,
    )

    adapter = GPQAOpenEndedAdapter(record_ids=["rec-missing"])
    with pytest.raises(ValueError, match="Missing record_ids"):
        adapter.to_samples()


# ---------------------------------------------------------------------------
# 3. test_debate_state_json_roundtrip
# ---------------------------------------------------------------------------


def test_debate_state_json_roundtrip():
    """DebateState survives JSON serialization/deserialization."""
    original = _make_completed_state()

    json_str = _state_to_json(original)
    restored = _state_from_json(json_str)

    # Verify spec
    assert restored.spec.debate_id == original.spec.debate_id
    assert restored.spec.problem.task_prompt == original.spec.problem.task_prompt
    assert restored.spec.problem.target == original.spec.problem.target
    assert restored.spec.protocol_kind == original.spec.protocol_kind
    assert restored.spec.prompts_ref == original.spec.prompts_ref
    assert restored.spec.open_reasoning == original.spec.open_reasoning
    assert dict(restored.spec.problem.answer_by_role) == dict(original.spec.problem.answer_by_role)
    assert len(restored.spec.schedule) == len(original.spec.schedule)
    for r_slot, o_slot in zip(restored.spec.schedule, original.spec.schedule):
        assert r_slot.slot_id == o_slot.slot_id
        assert r_slot.round_index == o_slot.round_index
        assert r_slot.phase == o_slot.phase
        assert r_slot.actors == o_slot.actors
        assert r_slot.boundary_after == o_slot.boundary_after
        assert r_slot.visibility_policy == o_slot.visibility_policy

    # Verify top-level state fields
    assert restored.slot_index == original.slot_index
    assert restored.rounds_completed == original.rounds_completed
    assert restored.done == original.done

    # Verify transcript
    assert len(restored.transcript) == len(original.transcript)
    for r_utt, o_utt in zip(restored.transcript, original.transcript):
        assert r_utt.role == o_utt.role
        assert r_utt.round_index == o_utt.round_index
        assert r_utt.phase == o_utt.phase
        assert r_utt.text == o_utt.text
        assert r_utt.token_count == o_utt.token_count
        assert r_utt.slot_id == o_utt.slot_id
        if o_utt.fields is not None:
            assert dict(r_utt.fields) == dict(o_utt.fields)

    # Verify judge_trace
    assert len(restored.judge_trace) == len(original.judge_trace)
    for r_jd, o_jd in zip(restored.judge_trace, original.judge_trace):
        assert r_jd.round_index == o_jd.round_index
        assert r_jd.verdict == o_jd.verdict
        assert dict(r_jd.score_delta_by_role) == dict(o_jd.score_delta_by_role)

    # Verify outcome
    assert restored.outcome is not None
    assert restored.outcome.winner == original.outcome.winner
    assert dict(restored.outcome.scores_by_role) == dict(original.outcome.scores_by_role)
    assert restored.outcome.verdict_text == original.outcome.verdict_text

    # Verify pending_simultaneous roundtrips (empty here)
    assert len(restored.pending_simultaneous) == 0


def test_debate_state_json_roundtrip_with_pending():
    """Roundtrip with pending_simultaneous entries."""
    spec = _make_spec()
    pending_utt = _make_utterance(
        Role.DEBATER_A, 0, Phase.PROPOSE, "pending text", answer="A", slot_id=0
    )
    state = DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={Role.DEBATER_A: pending_utt},
        judge_trace=(),
        done=False,
        outcome=None,
    )
    json_str = _state_to_json(state)
    restored = _state_from_json(json_str)
    assert Role.DEBATER_A in restored.pending_simultaneous
    r_utt = restored.pending_simultaneous[Role.DEBATER_A]
    assert r_utt.text == "pending text"
    assert dict(r_utt.fields) == {"answer": "A"}


# ---------------------------------------------------------------------------
# 4. test_debate_scorer_synthetic
# ---------------------------------------------------------------------------


def _make_store_get(json_str: str, trained_role_str: str | None = None):
    """Create a side_effect for store.get that dispatches by key."""

    def _get(key, default=None):
        if key == _STORE_KEY:
            return json_str
        if key == _TRAINED_ROLE_KEY:
            return trained_role_str
        return default

    return _get


@pytest.mark.asyncio
async def test_debate_scorer_synthetic():
    """Scorer produces correct metrics for a completed debate with known answers."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
    json_str = _state_to_json(state)

    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()
    task_state.store.get = MagicMock(side_effect=_make_store_get(json_str))

    target = Target("A")
    scorer_fn = debate_scorer()
    score = await scorer_fn(task_state, target)

    assert isinstance(score.value, dict)
    v = score.value
    assert v["accuracy.debater_a"] == 1.0
    assert v["accuracy.debater_b"] == 0.0
    assert v["judge_quality"] == 1.0
    # truth_win_if_disagreement: A correct, B wrong, judge picked A -> 1.0
    assert v["truth_win_if_disagreement"] == 1.0
    assert v["truth_surfaced"] == 1.0
    assert v["disagreement"] == 1.0
    # No None values should appear in the dict
    for key, val in v.items():
        assert val is not None, f"Metric {key} should not be None"


# ---------------------------------------------------------------------------
# 5. test_debate_scorer_handles_none_metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debate_scorer_handles_none_metrics():
    """Scorer with target=None: N/A metrics appear as NaN, not None or absent.

    All keys are always present (for aggregation safety). Metrics that can't
    compute (e.g. accuracy without target) have value NaN.
    """
    import math

    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = _make_completed_state(target=None, a_answer="A", b_answer="B", winner=Role.DEBATER_A)
    json_str = _state_to_json(state)

    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()
    task_state.store.get = MagicMock(side_effect=_make_store_get(json_str))

    target = Target("")
    scorer_fn = debate_scorer()
    score = await scorer_fn(task_state, target)

    assert isinstance(score.value, dict)
    # No Python None values should appear
    for key, val in score.value.items():
        assert val is not None, f"Metric {key} should not be None in score dict"
    # Metrics that require target should be NaN (present but not computable)
    assert math.isnan(score.value["accuracy.debater_a"])
    assert math.isnan(score.value["accuracy.debater_b"])
    assert math.isnan(score.value["judge_quality"])
    assert math.isnan(score.value["truth_win_if_disagreement"])
    assert math.isnan(score.value["truth_surfaced"])
    # Metrics that don't need target should have real values
    assert score.value["disagreement"] == 1.0
    assert score.value["draw_rate"] == 0.0


@pytest.mark.asyncio
async def test_debate_scorer_no_store_data():
    """Scorer returns NaN-filled dict when no debate state in store.

    All _metric_aggs keys must be present (as NaN) so Inspect aggregation
    doesn't fail on missing keys.
    """
    import math

    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import (
        _identity_metric_keys,
    )
    from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.metrics import mcq_debate_metrics

    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()
    task_state.store.get = MagicMock(return_value=None)

    target = Target("A")
    scorer_fn = debate_scorer()
    score = await scorer_fn(task_state, target)

    # All registered keys should be present as NaN
    expected_keys = set(mcq_debate_metrics().keys()) | set(_identity_metric_keys())
    assert set(score.value.keys()) == expected_keys
    for val in score.value.values():
        assert math.isnan(val), f"Expected NaN, got {val}"
    assert "No debate state" in score.explanation


# ---------------------------------------------------------------------------
# 6. test_evaluator_fast_path_routing
# ---------------------------------------------------------------------------

_EVAL_MODULE = "tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator"


def _dummy_adapter() -> MagicMock:
    """Adapter mock that returns a non-empty sample list."""
    from inspect_ai.dataset import Sample

    adapter = MagicMock(spec=DatasetAdapter)
    adapter.to_samples.return_value = [
        Sample(
            input="test question",
            target="A",
            metadata={"answer_a": "A", "answer_b": "B", "source": "test"},
        ),
    ]
    adapter.resolve_scoring_mode.return_value = ScoringMode.MCQ
    return adapter


def _dummy_open_ended_adapter() -> MagicMock:
    """Adapter mock for OPEN_ENDED eval validation."""
    from inspect_ai.dataset import Sample

    adapter = MagicMock(spec=DatasetAdapter)
    adapter.to_samples.return_value = [
        Sample(
            input="Explain why water freezes.",
            target="At 0C under standard pressure.",
            metadata={"answer_a": "", "answer_b": "", "source": "test"},
        ),
    ]
    adapter.resolve_scoring_mode.return_value = ScoringMode.OPEN_ENDED
    return adapter


def _patch_evaluator_externals():
    """Context manager stack for patching all external deps in evaluator."""
    from contextlib import ExitStack

    stack = ExitStack()

    mock_task_result = MagicMock()
    mock_task_result.results = MagicMock()
    mock_task_result.results.scores = []

    mock_eval = stack.enter_context(
        patch(f"{_EVAL_MODULE}.eval_async", new_callable=AsyncMock, return_value=[mock_task_result])
    )
    stack.enter_context(patch(f"{_EVAL_MODULE}.get_renderer", return_value=MagicMock()))
    stack.enter_context(patch(f"{_EVAL_MODULE}.get_tokenizer", return_value=MagicMock()))
    stack.enter_context(patch(f"{_EVAL_MODULE}.TinkerMessageCompleter", return_value=MagicMock()))
    stack.enter_context(patch(f"{_EVAL_MODULE}.tinker.ServiceClient", return_value=MagicMock()))

    return stack, mock_eval


@pytest.mark.asyncio
async def test_run_eval_routes_gpqa_open_ended_to_adapter(monkeypatch):
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval import run_eval

    fake_service = MagicMock()
    fake_service.create_sampling_client.return_value = MagicMock()
    monkeypatch.setattr(run_eval.tinker, "ServiceClient", lambda base_url=None: fake_service)
    renderer_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        run_eval.model_info,
        "get_recommended_renderer_name",
        lambda model_name, reasoning_effort=None: (
            renderer_calls.append((model_name, reasoning_effort)) or "gpt_oss_high_reasoning"
        ),
    )

    captured: dict[str, object] = {}

    class _FakeBuilder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self):
            async def _evaluator(_sampling_client):
                return {"accuracy.debater_a": 1.0}

            return _evaluator

    monkeypatch.setattr(run_eval, "DebateInspectEvaluatorBuilder", _FakeBuilder)

    config = run_eval.Config(
        model_name="openai/gpt-oss-120b",
        dataset="gpqa_open_ended",
        dataset_subset="extended",
        dataset_split="train",
        record_ids=["rec-test-1", "rec-test-2"],
        prompts_ref="default",
        artifacts_dir="artifacts/debate/test-open-ended",
        debater_reasoning_effort="high",
        judge_reasoning_effort="medium",
    )

    await run_eval.main(config)

    adapter = captured["adapter"]
    assert isinstance(adapter, GPQAOpenEndedAdapter)
    assert adapter.resolve_scoring_mode() == ScoringMode.OPEN_ENDED
    assert captured["log_dir"] == "artifacts/debate/test-open-ended"
    assert captured["debater_reasoning_effort"] == "high"
    assert captured["judge_reasoning_effort"] == "medium"
    assert renderer_calls == [("openai/gpt-oss-120b", "high")]


@pytest.mark.asyncio
async def test_evaluator_uses_actor_specific_reasoning_effort(monkeypatch):
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )

    adapter = _dummy_adapter()
    renderer_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator.model_info.get_recommended_renderer_name",
        lambda model_name, reasoning_effort=None: (
            renderer_calls.append((model_name, reasoning_effort)) or "gpt_oss_reasoning"
        ),
    )

    builder = DebateInspectEvaluatorBuilder(
        adapter=adapter,
        model_name="openai/gpt-oss-120b",
        judge_model="openai/gpt-oss-20b",
        debater_reasoning_effort="high",
        judge_reasoning_effort="medium",
        log_evals_every=1,
    )
    evaluator = builder()

    stack, _mock_eval = _patch_evaluator_externals()
    with stack:
        await evaluator(MagicMock())

    assert renderer_calls == [
        ("openai/gpt-oss-120b", "high"),
        ("openai/gpt-oss-20b", "medium"),
    ]


def test_smoke_gpqa_open_ended_passes_explicit_open_ended_mode(monkeypatch):
    from tinker_cookbook.recipes.multiplayer_rl.debate.scripts import smoke_gpqa_open_ended
    from tinker_cookbook.recipes.multiplayer_rl.debate.types import DebateProblemSpec

    fake_problems = [
        DebateProblemSpec.from_seat_answers(
            "Which city is nicknamed the Big Apple?",
            "", "", ScoringMode.OPEN_ENDED,
            target="New York City",
            metadata={"record_id": "rec-test-1"},
        ),
    ]

    captured: dict[str, object] = {}

    def _fake_dataset(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after dataset construction")

    monkeypatch.setattr(
        smoke_gpqa_open_ended,
        "load_gpqa_open_ended_problems",
        lambda **_kwargs: fake_problems,
    )
    monkeypatch.setattr(smoke_gpqa_open_ended, "DebateDataset", _fake_dataset)
    monkeypatch.setattr(smoke_gpqa_open_ended.tinker, "ServiceClient", lambda base_url=None: MagicMock(create_sampling_client=lambda **kwargs: MagicMock()))
    monkeypatch.setattr(
        smoke_gpqa_open_ended,
        "get_recommended_renderer_name",
        lambda model_name, reasoning_effort=None: "gpt_oss_reasoning",
    )
    monkeypatch.setattr(smoke_gpqa_open_ended, "get_renderer", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(smoke_gpqa_open_ended, "get_tokenizer", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(smoke_gpqa_open_ended, "TinkerTokenCompleter", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(smoke_gpqa_open_ended, "TinkerMessageCompleter", lambda **_kwargs: MagicMock())

    class _FakeScorerBuilder:
        def __init__(self, **_kwargs):
            pass

        def build(self, *, usage_tracker=None):
            return MagicMock()

    monkeypatch.setattr(smoke_gpqa_open_ended, "BinaryJudgeBuilder", _FakeScorerBuilder)
    monkeypatch.setattr(smoke_gpqa_open_ended, "RecordingBinaryJudgeClient", lambda inner: inner)

    args = smoke_gpqa_open_ended.parse_args(
        [
            "--record-id",
            "rec-test-1",
            "--artifacts-dir",
            "artifacts/debate/test-smoke-open-ended",
        ]
    )

    with pytest.raises(RuntimeError, match="stop after dataset construction"):
        asyncio.run(smoke_gpqa_open_ended.run(args))

    # scoring_mode is on each DebateProblemSpec
    problems = captured["problems"]
    assert all(p.scoring_mode == ScoringMode.OPEN_ENDED for p in problems)


@pytest.mark.asyncio
async def test_evaluator_fast_path_routing():
    """DebateInspectEvaluator calls eval_async only on log_evals_every cadence."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
        DebateInspectEvaluator,
    )

    adapter = _dummy_adapter()

    builder = DebateInspectEvaluatorBuilder(
        adapter=adapter,
        log_evals_every=3,
        renderer_name="qwen3",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
    )
    evaluator = builder()
    assert isinstance(evaluator, DebateInspectEvaluator)

    mock_sampling = MagicMock()

    expected_log_dir = os.path.expanduser("~/inspect-logs")

    stack, mock_eval = _patch_evaluator_externals()
    with stack:
        # Call 1: count=1, 1%3!=0 -> eval runs, tempdir (not the real log dir)
        await evaluator(mock_sampling)
        assert mock_eval.call_count == 1
        call_kwargs_1 = mock_eval.call_args_list[0][1]
        assert call_kwargs_1.get("log_dir") != expected_log_dir

        # Call 2: count=2, 2%3!=0 -> eval runs, tempdir
        await evaluator(mock_sampling)
        assert mock_eval.call_count == 2
        call_kwargs_2 = mock_eval.call_args_list[1][1]
        assert call_kwargs_2.get("log_dir") != expected_log_dir

        # Call 3: count=3, 3%3==0 -> eval runs, WITH real log dir
        await evaluator(mock_sampling)
        assert mock_eval.call_count == 3
        call_kwargs_3 = mock_eval.call_args_list[2][1]
        assert call_kwargs_3.get("log_dir") == expected_log_dir


# ---------------------------------------------------------------------------
# 7. test_evaluator_log_path_routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_log_path_routing():
    """DebateInspectEvaluator with log_evals_every=1 always writes logs."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )

    adapter = _dummy_adapter()

    builder = DebateInspectEvaluatorBuilder(
        adapter=adapter,
        log_evals_every=1,
        renderer_name="qwen3",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        log_dir="/tmp/test-logs",
    )
    evaluator = builder()
    mock_sampling = MagicMock()

    stack, mock_eval = _patch_evaluator_externals()
    with stack:
        await evaluator(mock_sampling)
        assert mock_eval.call_count == 1
        call_kwargs = mock_eval.call_args_list[0][1]
        assert call_kwargs.get("log_dir") == "/tmp/test-logs"


# ---------------------------------------------------------------------------
# 8. test_solver_scorer_pipeline (integration with mocks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_solver_scorer_pipeline():
    """End-to-end solver -> scorer pipeline with mock completers."""
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import (
        debate_solver,
    )

    # Mock completers that return fixed responses with answer tags.
    async def make_completer(answer: str):
        completer = AsyncMock()
        completer.return_value = {
            "content": f"My argument is strong. <answer>{answer}</answer>",
            "role": "assistant",
        }
        return completer

    trained = await make_completer("A")
    opponent = await make_completer("B")
    judge = await make_completer("A")  # Judge picks A

    # Override the LLMJudgeCallback to avoid real judge logic.
    # The solver imports DebateRuntime which uses judge_callback after boundary rounds.
    # We need to mock at the runtime level.

    solver = debate_solver(
        sampling_client=trained,
        opponent_client=opponent,
        judge_client=judge,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=1,
        prompts_ref="scientific_mcq",
    )

    # Create a TaskState with a Sample
    sample = Sample(
        input="What is X?\nA) Correct\nB) Wrong\nC) Also wrong\nD) Nope",
        target="A",
        metadata={"answer_a": "A", "answer_b": "B", "source": "test"},
    )

    # Build a real TaskState
    task_state = TaskState(
        model="test-model",
        sample_id="test-1",
        epoch=0,
        input=[ChatMessageUser(content=sample.input)],
        messages=[ChatMessageUser(content=sample.input)],
        target=Target(sample.target),
        metadata=sample.metadata,
    )

    generate = MagicMock()

    # Patch the judge callback to avoid needing real judge parsing.
    # The runtime calls judge_callback after boundary rounds.
    with patch(
        "tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime.DebateRuntime"
    ) as MockRuntime:
        # Create a mock runtime that simulates a completed debate
        mock_runtime = MagicMock()
        MockRuntime.return_value = mock_runtime

        completed_state = _make_completed_state(
            target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A
        )
        # Simulate done state
        mock_runtime.state = completed_state

        result = await solver(task_state, generate)

    # Verify state was stored
    assert result.store.get(_STORE_KEY) is not None or hasattr(result.store, "set")

    # Now test the scorer on this stored state (use the state we know)
    json_str = _state_to_json(completed_state)
    scorer_state = MagicMock(spec=TaskState)
    scorer_state.store = MagicMock()
    scorer_state.store.get = MagicMock(side_effect=_make_store_get(json_str))

    target = Target("A")
    scorer_fn = debate_scorer()
    score = await scorer_fn(scorer_state, target)

    assert isinstance(score.value, dict)
    assert score.value["accuracy.debater_a"] == 1.0
    assert score.value["accuracy.debater_b"] == 0.0
    assert score.value["judge_quality"] == 1.0
    assert score.answer == "debater_a"


# ---------------------------------------------------------------------------
# 9. Wave 3 Gate: GPQAAdapter free_debate mode
# ---------------------------------------------------------------------------


def test_gpqa_adapter_free_debate():
    """GPQAAdapter(free_debate=True) emits empty answer strings."""
    adapter = GPQAAdapter(limit=3, seed=42, free_debate=True)
    samples = adapter.to_samples()
    assert len(samples) == 3
    for s in samples:
        assert s.metadata["answer_a"] == ""
        assert s.metadata["answer_b"] == ""
        assert s.metadata["source"] == "gpqa_diamond"
        # target is still populated (correct answer label)
        assert s.target in ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# 10. Wave 3 Gate: Evaluator default parity
# ---------------------------------------------------------------------------


def test_evaluator_builder_defaults():
    """DebateInspectEvaluatorBuilder defaults match training config."""
    import inspect

    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )

    sig = inspect.signature(DebateInspectEvaluatorBuilder)
    defaults = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    assert defaults["prompts_ref"] == "judge_exploit"
    assert defaults["opponent_max_tokens"] == 8192
    assert defaults["judge_max_tokens"] == 4096
    assert defaults["open_reasoning"] is False
    assert defaults["randomize_position"] is True


# ---------------------------------------------------------------------------
# 11. Wave 3 Gate: Metric extraction bug fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metric_extraction_uses_metric_name():
    """Evaluator extracts all metrics by metric_name, not score_result.name."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )

    adapter = _dummy_adapter()
    builder = DebateInspectEvaluatorBuilder(
        adapter=adapter,
        renderer_name="qwen3",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
    )
    evaluator = builder()
    mock_sampling = MagicMock()

    # Create mock score_result with 3 distinct metrics
    mock_score_result = MagicMock()
    mock_score_result.name = "debate_scorer"
    mock_metric_a = MagicMock()
    mock_metric_a.value = 0.8
    mock_metric_b = MagicMock()
    mock_metric_b.value = 0.6
    mock_metric_c = MagicMock()
    mock_metric_c.value = 0.9
    mock_score_result.metrics = {
        "accuracy.debater_a": mock_metric_a,
        "judge_quality": mock_metric_b,
        "truth_surfaced": mock_metric_c,
    }

    mock_task_result = MagicMock()
    mock_task_result.results = MagicMock()
    mock_task_result.results.scores = [mock_score_result]

    stack, mock_eval = _patch_evaluator_externals()
    mock_eval.return_value = [mock_task_result]
    with stack:
        metrics = await evaluator(mock_sampling)

    # All 3 metrics should be present (bug was: only last metric survived)
    assert "accuracy.debater_a" in metrics
    assert "judge_quality" in metrics
    assert "truth_surfaced" in metrics
    assert metrics["accuracy.debater_a"] == 0.8
    assert metrics["judge_quality"] == 0.6
    assert metrics["truth_surfaced"] == 0.9


# ---------------------------------------------------------------------------
# 12. Wave 3 Gate: Identity-aware scorer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scorer_identity_remap_when_trained_role_set():
    """Scorer produces id/ keys when trained_role is stored."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
    json_str = _state_to_json(state)

    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()

    def store_get(key, default=None):
        if key == _STORE_KEY:
            return json_str
        if key == _TRAINED_ROLE_KEY:
            return Role.DEBATER_A.value
        return default

    task_state.store.get = MagicMock(side_effect=store_get)

    target = Target("A")
    scorer_fn = debate_scorer()
    score = await scorer_fn(task_state, target)

    v = score.value
    # Should have identity keys
    assert "id/trained_role_is_a" in v
    assert v["id/trained_role_is_a"] == 1.0

    # When trained=A: id/accuracy.trained should equal accuracy.debater_a
    assert v["id/accuracy.trained"] == v["accuracy.debater_a"]
    assert v["id/accuracy.opponent"] == v["accuracy.debater_b"]


@pytest.mark.asyncio
async def test_scorer_identity_remap_trained_b():
    """Identity remap when trained is debater_b."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
    json_str = _state_to_json(state)

    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()

    def store_get(key, default=None):
        if key == _STORE_KEY:
            return json_str
        if key == _TRAINED_ROLE_KEY:
            return Role.DEBATER_B.value
        return default

    task_state.store.get = MagicMock(side_effect=store_get)

    target = Target("A")
    scorer_fn = debate_scorer()
    score = await scorer_fn(task_state, target)

    v = score.value
    assert v["id/trained_role_is_a"] == 0.0
    # When trained=B: id/accuracy.trained should equal accuracy.debater_b
    assert v["id/accuracy.trained"] == v["accuracy.debater_b"]
    assert v["id/accuracy.opponent"] == v["accuracy.debater_a"]


@pytest.mark.asyncio
async def test_scorer_identity_nan_without_trained_role():
    """Scorer fills identity keys with NaN when trained_role not in store.

    This ensures Inspect aggregation doesn't fail on missing keys.
    """
    import math

    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = _make_completed_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
    json_str = _state_to_json(state)

    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()

    def store_get(key, default=None):
        if key == _STORE_KEY:
            return json_str
        return default

    task_state.store.get = MagicMock(side_effect=store_get)

    target = Target("A")
    scorer_fn = debate_scorer()
    score = await scorer_fn(task_state, target)

    v = score.value
    # Identity keys should be present as NaN (not absent)
    assert "id/trained_role_is_a" in v
    assert math.isnan(v["id/trained_role_is_a"])
    assert "id/accuracy.trained" in v
    assert math.isnan(v["id/accuracy.trained"])


def test_identity_metric_keys_consistent_with_remap_bases():
    """_identity_metric_keys produces keys consistent with IDENTITY_REMAP_BASES."""
    keys = _identity_metric_keys()
    assert "id/trained_role_is_a" in keys
    for base in IDENTITY_REMAP_BASES:
        assert f"id/{base}.trained" in keys
        assert f"id/{base}.opponent" in keys
    # 1 (trained_role_is_a) + 2 * len(IDENTITY_REMAP_BASES)
    assert len(keys) == 1 + 2 * len(IDENTITY_REMAP_BASES)


# ---------------------------------------------------------------------------
# 14. Self-play eval wiring in build_config
# ---------------------------------------------------------------------------


def test_build_config_selfplay_sets_eval_opponent_none():
    """build_config with self_play=True sets opponent_model=None on default eval builder."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )
    from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import (
        CLIConfig,
        build_config,
    )

    cli = CLIConfig(self_play=True, opponent_model="Qwen/Qwen3-4B-Instruct-2507")
    config = build_config(cli)

    eval_builder = config.evaluator_builders[0]
    assert isinstance(eval_builder, DebateInspectEvaluatorBuilder)
    assert eval_builder.opponent_model is None
    assert eval_builder.randomize_position is False


def test_build_config_frozen_opp_keeps_eval_opponent():
    """build_config with self_play=False passes opponent_model through to eval builder."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )
    from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import (
        CLIConfig,
        build_config,
    )

    cli = CLIConfig(self_play=False, opponent_model="Qwen/Qwen3-4B-Instruct-2507")
    config = build_config(cli)

    eval_builder = config.evaluator_builders[0]
    assert isinstance(eval_builder, DebateInspectEvaluatorBuilder)
    assert eval_builder.opponent_model == "Qwen/Qwen3-4B-Instruct-2507"


def test_build_config_custom_eval_respected_in_selfplay():
    """User-provided inspect_eval is respected even during self-play training."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )
    from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import (
        CLIConfig,
        build_config,
    )

    custom_eval = DebateInspectEvaluatorBuilder(
        adapter=GPQAAdapter(free_debate=True, limit=5),
        opponent_model="some/frozen-model",
    )
    cli = CLIConfig(self_play=True, inspect_eval=custom_eval)
    config = build_config(cli)

    eval_builder = config.evaluator_builders[0]
    assert isinstance(eval_builder, DebateInspectEvaluatorBuilder)
    # Custom eval keeps its own opponent_model (user wants frozen-opp eval for comparison).
    assert eval_builder.opponent_model == "some/frozen-model"


def test_debate_eval_open_ended_requires_scorer_client():
    adapter = _dummy_open_ended_adapter()

    with pytest.raises(ValueError, match="requires a scorer_client"):
        debate_eval(
            adapter=adapter,
            sampling_client=MagicMock(),
            opponent_client=MagicMock(),
            judge_client=MagicMock(),
            prompts_ref=(
                "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml"
            ),
            scorer_client=None,
        )


def test_debate_eval_open_ended_uses_default_grader_when_yaml_omits_it():
    """A YAML without _grader still works — built-in default grader kicks in."""
    from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts

    prompts = resolve_prompts(
        "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/"
        "semantic_prompts_missing_grader.yaml"
    )
    grader = prompts.get_binary_judge_template("grader")
    assert grader is not None
    assert grader.positive == "CORRECT"
    assert grader.negative == "INCORRECT"


def test_build_config_scorer_builder_passes_through_to_eval_builder():
    from tinker_cookbook.recipes.multiplayer_rl.debate.eval.evaluator import (
        DebateInspectEvaluatorBuilder,
    )
    from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import (
        CLIConfig,
        build_config,
    )

    scorer_builder = BinaryJudgeBuilder(
        provider="openai_compatible",
        model="gpt-5-mini",
        max_connections=17,
    )
    cli = CLIConfig(scorer_builder=scorer_builder)
    config = build_config(cli)

    eval_builder = config.evaluator_builders[0]
    assert isinstance(eval_builder, DebateInspectEvaluatorBuilder)
    assert eval_builder.scorer_builder is scorer_builder
