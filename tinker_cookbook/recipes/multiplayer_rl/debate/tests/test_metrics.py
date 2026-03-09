"""Comprehensive tests for debate metrics."""

from __future__ import annotations

from types import MappingProxyType

from ..scoring.metrics import (
    MetricResult,
    accuracy,
    choice_match,
    concession_correctness,
    convergence_round,
    debater_accuracy_delta,
    disagreement,
    draw_rate,
    exact_match,
    judge_quality,
    mcq_debate_metrics,
    parse_success,
    stance_change,
    truth_surfaced,
    truth_win_if_disagreement,
)
from ..types import (
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    Role,
    ScoringMode,
    TurnSlot,
    Utterance,
    VisibilityPolicy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _slot(slot_id: int, round_index: int, phase: Phase, role: Role) -> TurnSlot:
    return TurnSlot(
        slot_id=slot_id,
        round_index=round_index,
        phase=phase,
        actors=(role,),
        boundary_after=False,
        visibility_policy=VisibilityPolicy.ALL_PRIOR,
    )


def _schedule_2_rounds() -> tuple[TurnSlot, ...]:
    """2 rounds, propose+critique for each debater per round."""
    return (
        _slot(0, 0, Phase.PROPOSE, Role.DEBATER_A),
        _slot(1, 0, Phase.PROPOSE, Role.DEBATER_B),
        _slot(2, 0, Phase.CRITIQUE, Role.DEBATER_A),
        _slot(3, 0, Phase.CRITIQUE, Role.DEBATER_B),
        _slot(4, 1, Phase.PROPOSE, Role.DEBATER_A),
        _slot(5, 1, Phase.PROPOSE, Role.DEBATER_B),
        _slot(6, 1, Phase.CRITIQUE, Role.DEBATER_A),
        _slot(7, 1, Phase.CRITIQUE, Role.DEBATER_B),
    )


def _utt(
    role: Role,
    round_index: int,
    phase: Phase,
    text: str,
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


def _spec(target: str | None = "A") -> DebateSpec:
    return DebateSpec(
        debate_id="test-001",
        problem=DebateProblemSpec(
            task_prompt="Which option is correct?",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role=None,
            target=target,
        ),
        schedule=_schedule_2_rounds(),
        open_reasoning=False,
    )


def _outcome(winner: Role | None, scores: dict[Role, float] | None = None) -> DebateOutcome:
    if scores is None:
        scores = {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
    return DebateOutcome(winner=winner, scores_by_role=scores)


def _state(
    transcript: tuple[Utterance, ...],
    target: str | None = "A",
    outcome: DebateOutcome | None = None,
) -> DebateState:
    return DebateState(
        spec=_spec(target=target),
        slot_index=len(transcript),
        rounds_completed=2,
        transcript=transcript,
        pending_simultaneous=MappingProxyType({}),
        judge_trace=(),
        done=True,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


def test_exact_match():
    assert exact_match("A", "a")
    assert exact_match(" A ", "a")
    assert not exact_match("A", "B")


def test_choice_match():
    assert choice_match("The answer is A", "A")
    assert choice_match("A", "a")
    assert not choice_match("A", "B")
    assert not choice_match("not sure about A or B", "A")  # hedge → None


# ---------------------------------------------------------------------------
# Both debaters correct
# ---------------------------------------------------------------------------


def _both_correct_transcript() -> tuple[Utterance, ...]:
    return (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "I think A", answer="A"),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "I think A", answer="A"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "Still A", answer="A"),
        _utt(Role.DEBATER_B, 1, Phase.PROPOSE, "Still A", answer="A"),
    )


def test_both_correct_judge_picks_a():
    s = _state(_both_correct_transcript(), outcome=_outcome(Role.DEBATER_A))
    assert judge_quality()(s).value == 1.0


def test_both_correct_judge_picks_b():
    s = _state(_both_correct_transcript(), outcome=_outcome(Role.DEBATER_B))
    assert judge_quality()(s).value == 1.0


def test_both_correct_tie():
    s = _state(_both_correct_transcript(), outcome=_outcome(None))
    assert judge_quality()(s).value == 1.0


# ---------------------------------------------------------------------------
# A correct only
# ---------------------------------------------------------------------------


def _a_correct_only() -> tuple[Utterance, ...]:
    return (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "I think A", answer="A"),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "I think B", answer="B"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "Still A", answer="A"),
        _utt(Role.DEBATER_B, 1, Phase.PROPOSE, "Still B", answer="B"),
    )


def test_a_correct_judge_picks_a():
    s = _state(_a_correct_only(), outcome=_outcome(Role.DEBATER_A))
    assert judge_quality()(s).value == 1.0
    assert truth_win_if_disagreement()(s).value == 1.0


def test_a_correct_judge_picks_b():
    s = _state(_a_correct_only(), outcome=_outcome(Role.DEBATER_B))
    assert judge_quality()(s).value == 0.0
    assert truth_win_if_disagreement()(s).value == 0.0


def test_a_correct_judge_tie():
    s = _state(_a_correct_only(), outcome=_outcome(None))
    assert judge_quality()(s).value == 0.0
    assert truth_win_if_disagreement()(s).value == 0.0


# ---------------------------------------------------------------------------
# Both wrong
# ---------------------------------------------------------------------------


def _both_wrong() -> tuple[Utterance, ...]:
    return (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "I think B", answer="B"),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "I think C", answer="C"),
    )


def test_both_wrong_judge_quality():
    # Target is A, neither is correct
    s = _state(_both_wrong(), outcome=_outcome(Role.DEBATER_A))
    assert judge_quality()(s).value == 0.0
    s2 = _state(_both_wrong(), outcome=_outcome(Role.DEBATER_B))
    assert judge_quality()(s2).value == 0.0
    s3 = _state(_both_wrong(), outcome=_outcome(None))
    assert judge_quality()(s3).value == 0.0


def test_both_wrong_truth_win_none():
    # Both wrong → not exactly one correct → N/A
    s = _state(_both_wrong(), outcome=_outcome(Role.DEBATER_A))
    assert truth_win_if_disagreement()(s).value is None


# ---------------------------------------------------------------------------
# No target
# ---------------------------------------------------------------------------


def test_no_target_returns_none():
    transcript = _a_correct_only()
    s = _state(transcript, target=None, outcome=_outcome(Role.DEBATER_A))
    assert accuracy(Role.DEBATER_A)(s).value is None
    assert judge_quality()(s).value is None
    assert truth_win_if_disagreement()(s).value is None
    assert truth_surfaced()(s).value is None
    assert concession_correctness(Role.DEBATER_A)(s).value is None
    assert debater_accuracy_delta(Role.DEBATER_A)(s).value is None


# ---------------------------------------------------------------------------
# No outcome
# ---------------------------------------------------------------------------


def test_no_outcome_returns_none():
    s = _state(_a_correct_only(), outcome=None)
    assert judge_quality()(s).value is None
    assert truth_win_if_disagreement()(s).value is None
    assert draw_rate()(s).value is None


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_accuracy():
    s = _state(_a_correct_only(), outcome=_outcome(Role.DEBATER_A))
    assert accuracy(Role.DEBATER_A)(s).value == 1.0
    assert accuracy(Role.DEBATER_B)(s).value == 0.0


# ---------------------------------------------------------------------------
# Truth surfaced
# ---------------------------------------------------------------------------


def test_truth_surfaced_one_correct():
    s = _state(_a_correct_only())
    assert truth_surfaced()(s).value == 1.0


def test_truth_surfaced_none_correct():
    s = _state(_both_wrong())
    assert truth_surfaced()(s).value == 0.0


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


def test_disagreement_true():
    s = _state(_a_correct_only())
    assert disagreement()(s).value == 1.0


def test_disagreement_false():
    s = _state(_both_correct_transcript())
    assert disagreement()(s).value == 0.0


def test_disagreement_missing_answer():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "no answer here"),)
    s = _state(transcript)
    assert disagreement()(s).value is None


# ---------------------------------------------------------------------------
# Stance change
# ---------------------------------------------------------------------------


def _stance_change_transcript() -> tuple[Utterance, ...]:
    return (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "I think A", answer="A"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "Actually B", answer="B"),
    )


def test_stance_change():
    s = _state(_stance_change_transcript())
    assert stance_change(Role.DEBATER_A)(s).value == 1.0


def test_no_stance_change():
    s = _state(_both_correct_transcript())
    assert stance_change(Role.DEBATER_A)(s).value == 0.0


def test_stance_change_single_answer():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),)
    s = _state(transcript)
    assert stance_change(Role.DEBATER_A)(s).value is None


# ---------------------------------------------------------------------------
# Concession correctness
# ---------------------------------------------------------------------------


def test_correct_concession():
    """First wrong, last correct → +1.0 (genuine revision)."""
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "B", answer="B"),  # wrong
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "A", answer="A"),  # correct
    )
    s = _state(transcript, target="A")
    assert concession_correctness(Role.DEBATER_A)(s).value == 1.0


def test_sycophantic_concession():
    """First correct, last wrong → -1.0 (sycophantic capitulation)."""
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),  # correct
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "B", answer="B"),  # wrong
    )
    s = _state(transcript, target="A")
    assert concession_correctness(Role.DEBATER_A)(s).value == -1.0


def test_no_concession():
    """First == last → 0.0."""
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "A", answer="A"),
    )
    s = _state(transcript, target="A")
    assert concession_correctness(Role.DEBATER_A)(s).value == 0.0


# ---------------------------------------------------------------------------
# Debater accuracy delta
# ---------------------------------------------------------------------------


def test_accuracy_delta_positive():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "B", answer="B"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "A", answer="A"),
    )
    s = _state(transcript, target="A")
    assert debater_accuracy_delta(Role.DEBATER_A)(s).value == 1.0


def test_accuracy_delta_negative():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "B", answer="B"),
    )
    s = _state(transcript, target="A")
    assert debater_accuracy_delta(Role.DEBATER_A)(s).value == -1.0


def test_accuracy_delta_zero():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "A", answer="A"),
    )
    s = _state(transcript, target="A")
    assert debater_accuracy_delta(Role.DEBATER_A)(s).value == 0.0


# ---------------------------------------------------------------------------
# Convergence round
# ---------------------------------------------------------------------------


def test_convergence_round_immediate():
    s = _state(_both_correct_transcript())
    assert convergence_round()(s).value == 0.0


def test_convergence_round_later():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="B"),
        _utt(Role.DEBATER_A, 1, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_B, 1, Phase.PROPOSE, "A", answer="A"),
    )
    s = _state(transcript)
    assert convergence_round()(s).value == 1.0


def test_convergence_round_never():
    s = _state(_a_correct_only())
    assert convergence_round()(s).value is None


# ---------------------------------------------------------------------------
# Draw rate
# ---------------------------------------------------------------------------


def test_draw_rate_tie():
    s = _state(_a_correct_only(), outcome=_outcome(None))
    assert draw_rate()(s).value == 1.0


def test_draw_rate_winner():
    s = _state(_a_correct_only(), outcome=_outcome(Role.DEBATER_A))
    assert draw_rate()(s).value == 0.0


# ---------------------------------------------------------------------------
# Parse success
# ---------------------------------------------------------------------------


def test_parse_success_all():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="B"),
    )
    s = _state(transcript)
    fn = parse_success(
        roles=(Role.DEBATER_A, Role.DEBATER_B),
        phases=(Phase.PROPOSE,),
    )
    assert fn(s).value == 1.0


def test_parse_success_partial():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "no answer"),
    )
    s = _state(transcript)
    fn = parse_success(
        roles=(Role.DEBATER_A, Role.DEBATER_B),
        phases=(Phase.PROPOSE,),
    )
    assert fn(s).value == 0.5


def test_parse_success_filters():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A"),
        _utt(Role.JUDGE, 0, Phase.JUDGE_VERDICT, "winner A"),  # filtered out
    )
    s = _state(transcript)
    fn = parse_success(roles=(Role.DEBATER_A,), phases=(Phase.PROPOSE,))
    assert fn(s).value == 1.0


# ---------------------------------------------------------------------------
# mcq_debate_metrics smoke test
# ---------------------------------------------------------------------------


def test_mcq_debate_metrics_keys():
    metrics = mcq_debate_metrics()
    assert "accuracy.debater_a" in metrics
    assert "truth_win_if_disagreement" in metrics
    assert "parse_success" in metrics
    assert len(metrics) == 39


def test_mcq_debate_metrics_callable():
    metrics = mcq_debate_metrics()
    s = _state(_a_correct_only(), outcome=_outcome(Role.DEBATER_A))
    for name, fn in metrics.items():
        result = fn(s)
        assert isinstance(result, MetricResult), f"{name} did not return MetricResult"
