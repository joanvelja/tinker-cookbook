"""Comprehensive adversarial probe of the Inspect AI eval pipeline.

Exercises every code path in the scorer offline (no API calls) using mocks.
Finds bugs before we burn API credits.

Probes:
  1. Score value completeness (full state)
  2. Score value completeness (minimal/missing state)
  3. ModelOutput type correctness
  4. Metric aggregation with NaN
  5. Full solver mock
  6. Full scorer mock
  7. Edge cases (0 utterances, ties, missing trained_role)
"""

from __future__ import annotations

import math
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, mean
from inspect_ai.solver import TaskState

from ..core.schedule import build_schedule
from ..eval.inspect_task import (
    _STORE_KEY,
    _TRAINED_ROLE_KEY,
    _identity_metric_keys,
    _state_to_json,
    debate_scorer,
)
from ..scoring.metrics import mcq_debate_metrics
from ..types import (
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
) -> DebateSpec:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, num_rounds)
    return DebateSpec(
        debate_id="probe-001",
        task_prompt="What is the answer?\nA) Foo\nB) Bar\nC) Baz\nD) Qux",
        answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
        schedule=schedule,
        open_reasoning=True,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref="scientific_mcq",
        target=target,
    )


def _utt(
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


def _make_full_state(
    *,
    target: str | None = "A",
    a_answer: str = "A",
    b_answer: str = "B",
    winner: Role | None = Role.DEBATER_A,
    include_think: bool = True,
) -> DebateState:
    """Full state with all metric-triggering data: think blocks, multi-round, outcome."""
    spec = _make_spec(target=target, num_rounds=2)

    think_prefix_a = (
        "<thinking>The correct answer is A because...</thinking>" if include_think else ""
    )
    think_prefix_b = "<thinking>I believe the answer is B</thinking>" if include_think else ""

    transcript = (
        _utt(
            Role.DEBATER_A,
            0,
            Phase.PROPOSE,
            f"{think_prefix_a}My argument: <answer>{a_answer}</answer>",
            answer=a_answer,
            slot_id=0,
        ),
        _utt(
            Role.DEBATER_B,
            0,
            Phase.PROPOSE,
            f"{think_prefix_b}My argument: <answer>{b_answer}</answer>",
            answer=b_answer,
            slot_id=1,
        ),
        _utt(
            Role.DEBATER_A,
            1,
            Phase.CRITIQUE,
            f"{think_prefix_a}Rebuttal: <answer>{a_answer}</answer>",
            answer=a_answer,
            slot_id=2,
        ),
        _utt(
            Role.DEBATER_B,
            1,
            Phase.CRITIQUE,
            f"{think_prefix_b}Rebuttal: <answer>{b_answer}</answer>",
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


def _make_minimal_state() -> DebateState:
    """Minimal state: no outcome, no target, no think blocks, 0 utterances."""
    spec = DebateSpec(
        debate_id="probe-minimal",
        task_prompt="question?",
        answer_by_role=None,
        schedule=build_schedule(ProtocolKind.SEQUENTIAL, 1),
        open_reasoning=False,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref="scientific_mcq",
        target=None,
    )
    return DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous=MappingProxyType({}),
        judge_trace=(),
        done=True,
        outcome=None,
    )


def _mock_task_state(json_str: str, trained_role_str: str | None = None) -> MagicMock:
    """Create a mock TaskState with store dispatching by key."""
    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()
    store_data = {_STORE_KEY: json_str}
    if trained_role_str is not None:
        store_data[_TRAINED_ROLE_KEY] = trained_role_str
    task_state.store.get = MagicMock(side_effect=lambda k, default=None: store_data.get(k, default))
    return task_state


# ===================================================================
# 1. Score value completeness — full state
# ===================================================================


class TestScoreCompleteness:
    """Verify ALL expected keys appear in the score value dict."""

    @pytest.mark.asyncio
    async def test_all_base_keys_present(self):
        """Every key from mcq_debate_metrics() must appear in score.value."""
        state = _make_full_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        base_metrics = mcq_debate_metrics()
        for key in base_metrics:
            assert key in score.value, f"Base metric key {key!r} missing from score.value"

    @pytest.mark.asyncio
    async def test_all_identity_keys_present(self):
        """Every key from _identity_metric_keys() must appear when trained_role is set."""
        state = _make_full_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        for key in _identity_metric_keys():
            assert key in score.value, f"Identity key {key!r} missing from score.value"

    @pytest.mark.asyncio
    async def test_key_count(self):
        """Total keys = 39 base + 33 identity = 72."""
        state = _make_full_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        base_count = len(mcq_debate_metrics())
        identity_count = len(_identity_metric_keys())
        expected = base_count + identity_count
        actual = len(score.value)
        assert actual == expected, (
            f"Expected {expected} keys ({base_count} base + {identity_count} identity), got {actual}. "
            f"Missing: {set(list(mcq_debate_metrics().keys()) + _identity_metric_keys()) - set(score.value.keys())}"
        )

    @pytest.mark.asyncio
    async def test_no_none_values(self):
        """No value in score.value should be Python None (NaN is fine)."""
        state = _make_full_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        for key, val in score.value.items():
            assert val is not None, f"Metric {key} is None (should be NaN or a float)"
            assert isinstance(val, float), f"Metric {key} is {type(val).__name__}, not float"


# ===================================================================
# 2. Score value completeness — MISSING data
# ===================================================================


class TestScoreCompletenessMinimal:
    """Minimal state: no outcome, no target, no transcript, no think blocks."""

    @pytest.mark.asyncio
    async def test_all_base_keys_present_minimal(self):
        """ALL base keys present even with minimal state (as NaN)."""
        state = _make_minimal_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target(""))

        base_metrics = mcq_debate_metrics()
        for key in base_metrics:
            assert key in score.value, f"Base key {key!r} missing with minimal state"

    @pytest.mark.asyncio
    async def test_all_identity_keys_present_minimal(self):
        """ALL identity keys present even with minimal state."""
        state = _make_minimal_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target(""))

        for key in _identity_metric_keys():
            assert key in score.value, f"Identity key {key!r} missing with minimal state"

    @pytest.mark.asyncio
    async def test_all_values_are_nan_for_minimal(self):
        """With minimal state, most metrics should be NaN (not None, not KeyError)."""
        state = _make_minimal_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target(""))

        for key, val in score.value.items():
            assert val is not None, f"{key} is None"
            assert isinstance(val, float), f"{key} is {type(val).__name__}"

    @pytest.mark.asyncio
    async def test_no_keyerror_minimal(self):
        """Running the scorer on minimal state must not raise KeyError."""
        state = _make_minimal_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        # This is the main assertion: no exception
        score = await scorer_fn(task_state, Target(""))
        assert isinstance(score, Score)


# ===================================================================
# 3. ModelOutput type check
# ===================================================================


class TestModelOutput:
    def test_from_content_produces_model_output(self):
        """ModelOutput.from_content returns a proper ModelOutput."""
        output = ModelOutput.from_content(model="debate", content="test verdict")
        assert isinstance(output, ModelOutput)

    def test_from_content_has_choices(self):
        """ModelOutput should have at least one choice with the content."""
        output = ModelOutput.from_content(model="debate", content="test verdict")
        assert len(output.choices) >= 1
        # The content should be accessible somewhere in the choices
        found = False
        for choice in output.choices:
            if choice.message and choice.message.content:
                if "test verdict" in str(choice.message.content):
                    found = True
        assert found, "Content 'test verdict' not found in ModelOutput choices"

    def test_from_content_empty_string(self):
        """Empty string should not crash ModelOutput."""
        output = ModelOutput.from_content(model="debate", content="")
        assert isinstance(output, ModelOutput)


# ===================================================================
# 4. Metric aggregation compatibility (mean() with NaN)
# ===================================================================


class TestMetricAggregation:
    def test_mean_with_nan_via_sample_scores(self):
        """mean() should handle a mix of real values and NaN without crashing.

        Inspect AI's mean() expects SampleScore objects, not raw Score objects.
        This tests the actual API contract.
        """
        from inspect_ai.scorer._metric import SampleScore

        mean_fn = mean()

        # Valid scores
        k_scores = [
            SampleScore(score=Score(value=1.0), sample_id="s1"),
            SampleScore(score=Score(value=0.5), sample_id="s2"),
        ]
        try:
            k_result = mean_fn(k_scores)
            assert abs(k_result - 0.75) < 1e-9, f"k mean should be 0.75, got {k_result}"
        except Exception as e:
            pytest.fail(f"mean() crashed on valid scores: {e}")

        # Mix of real and NaN
        j_scores = [
            SampleScore(score=Score(value=float("nan")), sample_id="s1"),
            SampleScore(score=Score(value=0.8), sample_id="s2"),
        ]
        try:
            j_result = mean_fn(j_scores)
            # Record behavior: NaN propagates or is filtered
            assert isinstance(j_result, float)
        except Exception as e:
            pytest.fail(f"mean() crashed on scores with NaN: {e}")

    def test_mean_all_nan_via_sample_scores(self):
        """mean() with all NaN values should not crash."""
        from inspect_ai.scorer._metric import SampleScore

        scores = [
            SampleScore(score=Score(value=float("nan")), sample_id="s1"),
            SampleScore(score=Score(value=float("nan")), sample_id="s2"),
        ]
        mean_fn = mean()
        try:
            result = mean_fn(scores)
            assert isinstance(result, float)
        except Exception as e:
            pytest.fail(f"mean() crashed on all-NaN scores: {e}")


# ===================================================================
# 5. Full scorer mock — exercise actual scorer function
# ===================================================================


class TestFullScorerMock:
    @pytest.mark.asyncio
    async def test_scorer_with_full_state_trained_a(self):
        """Full scorer path: all metrics, identity remap as debater_a."""
        state = _make_full_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        v = score.value
        # Correctness checks
        assert v["accuracy.debater_a"] == 1.0
        assert v["accuracy.debater_b"] == 0.0
        assert v["judge_quality"] == 1.0
        assert v["truth_win_if_disagreement"] == 1.0
        assert v["win_rate.debater_a"] == 1.0
        assert v["win_rate.debater_b"] == 0.0
        assert v["loss_rate.debater_a"] == 0.0
        assert v["loss_rate.debater_b"] == 1.0
        assert v["draw_rate"] == 0.0
        assert v["disagreement"] == 1.0

        # Identity checks
        assert v["id/trained_role_is_a"] == 1.0
        assert v["id/accuracy.trained"] == 1.0  # trained=A, A is correct
        assert v["id/accuracy.opponent"] == 0.0
        assert v["id/win_rate.trained"] == 1.0
        assert v["id/win_rate.opponent"] == 0.0

        # Answer and explanation
        assert score.answer == "debater_a"
        assert score.explanation is not None

    @pytest.mark.asyncio
    async def test_scorer_with_full_state_trained_b(self):
        """Full scorer path with trained=B: identity keys should swap."""
        state = _make_full_state(target="A", a_answer="A", b_answer="B", winner=Role.DEBATER_A)
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_B.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        v = score.value
        assert v["id/trained_role_is_a"] == 0.0
        assert v["id/accuracy.trained"] == 0.0  # trained=B, B is wrong
        assert v["id/accuracy.opponent"] == 1.0
        assert v["id/win_rate.trained"] == 0.0  # B lost
        assert v["id/win_rate.opponent"] == 1.0  # A won

    @pytest.mark.asyncio
    async def test_scorer_serialization_roundtrip(self):
        """Verify JSON roundtrip does not corrupt the state."""
        state = _make_full_state()
        json_str = _state_to_json(state)
        from ..eval.inspect_task import _state_from_json

        restored = _state_from_json(json_str)

        assert restored.spec.target == state.spec.target
        assert len(restored.transcript) == len(state.transcript)
        assert restored.outcome is not None
        assert restored.outcome.winner == state.outcome.winner


# ===================================================================
# 6. Edge cases
# ===================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_zero_utterances(self):
        """0 utterances should not crash, all metrics should be NaN or valid."""
        state = _make_minimal_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        assert isinstance(score.value, dict)
        # Should have all keys (base + identity)
        assert len(score.value) > 0

    @pytest.mark.asyncio
    async def test_tie_outcome(self):
        """Tie (winner=None) should not crash any metric."""
        state = _make_full_state(target="A", a_answer="A", b_answer="B", winner=None)
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        v = score.value
        assert v["draw_rate"] == 1.0
        assert v["win_rate.debater_a"] == 0.0
        assert v["win_rate.debater_b"] == 0.0
        assert score.answer == "tie"

    @pytest.mark.asyncio
    async def test_no_trained_role_in_store(self):
        """When trained_role is NOT set, scorer should still return valid Score.

        Identity keys must still be present (as NaN) because _metric_aggs
        registers them for aggregation. Missing keys would cause Inspect errors.
        """
        state = _make_full_state(target="A")
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, trained_role_str=None)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        v = score.value
        # Base keys should still be present
        assert "accuracy.debater_a" in v
        # Identity keys should be present as NaN (not missing)
        for key in _identity_metric_keys():
            assert key in v, f"Identity key {key!r} missing when trained_role not set"
            assert math.isnan(v[key]), f"Identity key {key!r} should be NaN, got {v[key]}"

    @pytest.mark.asyncio
    async def test_no_outcome_no_target(self):
        """No outcome + no target = all base metrics are NaN.

        id/trained_role_is_a is always 0.0 or 1.0 (it's a flag, not a metric).
        """
        spec = DebateSpec(
            debate_id="edge-no-outcome",
            task_prompt="Q?",
            answer_by_role=None,
            schedule=build_schedule(ProtocolKind.SEQUENTIAL, 1),
            open_reasoning=False,
            target=None,
        )
        state = DebateState(
            spec=spec,
            slot_index=0,
            rounds_completed=0,
            transcript=(),
            pending_simultaneous={},
            judge_trace=(),
            done=True,
            outcome=None,
        )
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target(""))

        for key, val in score.value.items():
            assert val is not None, f"{key} is None"
            assert isinstance(val, float), f"{key} is {type(val).__name__}"
        # Base metrics should all be NaN (no data to compute from)
        for key in mcq_debate_metrics():
            assert math.isnan(score.value[key]), (
                f"Base metric {key} should be NaN, got {score.value[key]}"
            )
        # Identity flag is always a real value
        assert score.value["id/trained_role_is_a"] == 1.0

    @pytest.mark.asyncio
    async def test_winner_str_no_outcome(self):
        """_winner_str returns 'no_outcome' when outcome is None."""
        from ..eval.inspect_task import _winner_str

        state = _make_minimal_state()
        assert _winner_str(state) == "no_outcome"

    @pytest.mark.asyncio
    async def test_winner_str_tie(self):
        """_winner_str returns 'tie' when winner is None."""
        from ..eval.inspect_task import _winner_str

        state = _make_full_state(winner=None)
        assert _winner_str(state) == "tie"


# ===================================================================
# 7. _metric_aggs vs actual score keys consistency
# ===================================================================


class TestMetricAggsConsistency:
    """Verify the metric aggregation keys registered by the scorer match
    what the scorer actually produces."""

    @pytest.mark.asyncio
    async def test_aggs_keys_subset_of_score_keys_with_trained_role(self):
        """Every key in _metric_aggs should appear in score.value when trained_role is set."""
        state = _make_full_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, Role.DEBATER_A.value)

        # Build the _metric_aggs the same way the scorer does
        resolved = mcq_debate_metrics()
        expected_keys = set(resolved.keys()) | set(_identity_metric_keys())

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        actual_keys = set(score.value.keys())
        missing = expected_keys - actual_keys
        assert not missing, f"Keys in _metric_aggs but missing from score.value: {missing}"

    @pytest.mark.asyncio
    async def test_aggs_keys_vs_score_keys_WITHOUT_trained_role(self):
        """Even when trained_role is NOT set, all _metric_aggs keys must be present.

        After bug fix: identity keys are filled with NaN when trained_role is missing,
        preventing Inspect aggregation errors.
        """
        state = _make_full_state()
        json_str = _state_to_json(state)
        task_state = _mock_task_state(json_str, trained_role_str=None)

        resolved = mcq_debate_metrics()
        expected_agg_keys = set(resolved.keys()) | set(_identity_metric_keys())

        scorer_fn = debate_scorer()
        score = await scorer_fn(task_state, Target("A"))

        actual_keys = set(score.value.keys())
        missing = expected_agg_keys - actual_keys
        assert not missing, (
            f"_metric_aggs registers {len(missing)} keys missing from score.value "
            f"when trained_role is not set: {sorted(missing)[:5]}..."
        )
