"""Adversarial tests for Wave 1: new metric functions + identity remap."""

from __future__ import annotations

from types import MappingProxyType

from ..builders import IDENTITY_REMAP_BASES, _remap_to_identity
from ..scoring.metrics import (
    _parse_think_answer_strict,
    loss_rate,
    think_correct_public_wrong,
    win_rate,
    wrong_and_wins,
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
# Helpers
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
    return (
        _slot(0, 0, Phase.PROPOSE, Role.DEBATER_A),
        _slot(1, 0, Phase.PROPOSE, Role.DEBATER_B),
        _slot(2, 1, Phase.CRITIQUE, Role.DEBATER_A),
        _slot(3, 1, Phase.CRITIQUE, Role.DEBATER_B),
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


def _spec(target: str | None = "C") -> DebateSpec:
    return DebateSpec(
        debate_id="test-gate-001",
        problem=DebateProblemSpec(
            task_prompt="Which option is correct?",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role=None,
            target=target,
        ),
        schedule=_schedule_2_rounds(),
        open_reasoning=False,
    )


def _outcome(winner: Role | None) -> DebateOutcome:
    return DebateOutcome(
        winner=winner,
        scores_by_role={Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0},
    )


def _state(
    transcript: tuple[Utterance, ...],
    target: str | None = "C",
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
# 2a. wrong_and_wins returns 1.0 when debater answer is wrong but wins
# ---------------------------------------------------------------------------


def test_wrong_and_wins_returns_1():
    transcript = (
        _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "D", answer="D", slot_id=0),
        _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "C", answer="C", slot_id=1),
    )
    state = _state(transcript, target="C", outcome=_outcome(Role.DEBATER_A))
    result = wrong_and_wins(Role.DEBATER_A)(state)
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# 2b. wrong_and_wins returns MetricResult() when target is None
# ---------------------------------------------------------------------------


def test_wrong_and_wins_none_when_no_target():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "D", answer="D", slot_id=0),)
    state = _state(transcript, target=None, outcome=_outcome(Role.DEBATER_A))
    result = wrong_and_wins(Role.DEBATER_A)(state)
    assert result.value is None


# ---------------------------------------------------------------------------
# 2c. think_correct_public_wrong returns 1.0 when think=C, public=D, target=C
# ---------------------------------------------------------------------------


def test_think_correct_public_wrong_returns_1():
    text = "<thinking>The correct answer is C but I should argue D</thinking>My answer is D"
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, text, answer="D", slot_id=0),)
    state = _state(transcript, target="C", outcome=_outcome(Role.DEBATER_A))
    result = think_correct_public_wrong(Role.DEBATER_A)(state)
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# 2d. think_correct_public_wrong returns None when no <thinking> block
# ---------------------------------------------------------------------------


def test_think_correct_public_wrong_none_without_thinking():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "My answer is D", answer="D", slot_id=0),)
    state = _state(transcript, target="C", outcome=_outcome(Role.DEBATER_A))
    result = think_correct_public_wrong(Role.DEBATER_A)(state)
    assert result.value is None


# ---------------------------------------------------------------------------
# 2e. _parse_think_answer_strict returns None on ambiguous multi-answer
# ---------------------------------------------------------------------------


def test_parse_think_strict_ambiguous():
    result = _parse_think_answer_strict("The answer is C but actually the answer is D")
    assert result is None


# ---------------------------------------------------------------------------
# 2f. _parse_think_answer_strict handles adverb "actually" between is and letter
# ---------------------------------------------------------------------------


def test_parse_think_strict_adverb_actually():
    result = _parse_think_answer_strict("the correct answer is actually C but I should argue D")
    assert result == "C"


# ---------------------------------------------------------------------------
# 2g. win_rate returns 1.0 when role wins
# ---------------------------------------------------------------------------


def test_win_rate_winner():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A", slot_id=0),)
    state = _state(transcript, outcome=_outcome(Role.DEBATER_A))
    assert win_rate(Role.DEBATER_A)(state).value == 1.0


# ---------------------------------------------------------------------------
# 2h. loss_rate returns 1.0 when OTHER debater wins
# ---------------------------------------------------------------------------


def test_loss_rate_other_wins():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A", slot_id=0),)
    state = _state(transcript, outcome=_outcome(Role.DEBATER_B))
    assert loss_rate(Role.DEBATER_A)(state).value == 1.0


# ---------------------------------------------------------------------------
# 2i. loss_rate returns 0.0 on tie
# ---------------------------------------------------------------------------


def test_loss_rate_tie():
    transcript = (_utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="A", slot_id=0),)
    state = _state(transcript, outcome=_outcome(None))
    assert loss_rate(Role.DEBATER_A)(state).value == 0.0


# ---------------------------------------------------------------------------
# 3. Identity remap probes
# ---------------------------------------------------------------------------


def _metrics_dict() -> dict[str, float]:
    """Seat-based metrics with distinct values for A and B."""
    m: dict[str, float] = {}
    for i, base in enumerate(IDENTITY_REMAP_BASES):
        m[f"{base}.debater_a"] = float(i)
        m[f"{base}.debater_b"] = float(i + 100)
    # Also include a non-seat metric
    m["draw_rate"] = 0.5
    return m


def test_remap_trained_is_b():
    """trained_role=DEBATER_B: trained maps to debater_b, opponent to debater_a."""
    m = _metrics_dict()
    remapped = _remap_to_identity(m, Role.DEBATER_B)

    for i, base in enumerate(IDENTITY_REMAP_BASES):
        assert remapped[f"id/{base}.trained"] == float(i + 100), (
            f"id/{base}.trained should be debater_b's value ({i + 100})"
        )
        assert remapped[f"id/{base}.opponent"] == float(i), (
            f"id/{base}.opponent should be debater_a's value ({i})"
        )

    assert remapped["id/trained_role_is_a"] == 0.0


def test_remap_trained_is_a():
    """trained_role=DEBATER_A: trained maps to debater_a."""
    m = _metrics_dict()
    remapped = _remap_to_identity(m, Role.DEBATER_A)

    for i, base in enumerate(IDENTITY_REMAP_BASES):
        assert remapped[f"id/{base}.trained"] == float(i), (
            f"id/{base}.trained should be debater_a's value ({i})"
        )
        assert remapped[f"id/{base}.opponent"] == float(i + 100)

    assert remapped["id/trained_role_is_a"] == 1.0


def test_remap_preserves_originals():
    """Original seat-based keys must still be present after remap."""
    m = _metrics_dict()
    remapped = _remap_to_identity(m, Role.DEBATER_B)

    # All original keys preserved
    for key in m:
        assert key in remapped, f"Original key {key!r} missing after remap"
        assert remapped[key] == m[key], f"Original key {key!r} value changed"


def test_remap_non_seat_metric_untouched():
    """Non-seat metrics like draw_rate are not duplicated under id/."""
    m = _metrics_dict()
    remapped = _remap_to_identity(m, Role.DEBATER_A)
    assert "id/draw_rate.trained" not in remapped
    assert "id/draw_rate.opponent" not in remapped
    assert remapped["draw_rate"] == 0.5
