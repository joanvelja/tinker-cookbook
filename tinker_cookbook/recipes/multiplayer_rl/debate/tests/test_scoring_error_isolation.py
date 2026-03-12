"""Tests for Bug 4: scoring error isolation and three-valued metric logic."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from ..scoring.facts import (
    BinaryJudgeError,
    ResolvedDebateFacts,
    _tolerant_extract,
    built_in_metric_values,
    resolve_debate_facts_for_states,
    _matcher_key,
    _grader_key,
)
from ..prompts import resolve_prompts
from ..scoring.providers import AnswerJudgeClient
from ..types import (
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
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


def _schedule_3_rounds() -> tuple[TurnSlot, ...]:
    return (
        _slot(0, 0, Phase.PROPOSE, Role.DEBATER_A),
        _slot(1, 0, Phase.PROPOSE, Role.DEBATER_B),
        _slot(2, 1, Phase.CRITIQUE, Role.DEBATER_A),
        _slot(3, 1, Phase.CRITIQUE, Role.DEBATER_B),
        _slot(4, 2, Phase.CRITIQUE, Role.DEBATER_A),
        _slot(5, 2, Phase.CRITIQUE, Role.DEBATER_B),
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


def _state(
    *,
    target: str | None,
    scoring_mode: ScoringMode,
    transcript: tuple[Utterance, ...],
    winner: Role | None = Role.DEBATER_A,
    schedule: tuple[TurnSlot, ...] | None = None,
) -> DebateState:
    if schedule is None:
        schedule = _schedule_3_rounds()
    outcome = DebateOutcome(
        winner=winner,
        scores_by_role={Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0},
    )
    spec = DebateSpec(
        debate_id="test-bug4",
        problem=DebateProblemSpec(
            task_prompt="What is 2+2?",
            scoring_mode=scoring_mode,
            answer_by_role=None,
            target=target,
        ),
        schedule=schedule,
        open_reasoning=False,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref="default",
    )
    return DebateState(
        spec=spec,
        slot_index=len(transcript),
        rounds_completed=3,
        transcript=transcript,
        pending_simultaneous={},
        judge_trace=(),
        done=True,
        outcome=outcome,
    )


def _make_facts(
    scoring_mode: ScoringMode,
    equivalence: dict,
    correctness: dict,
) -> ResolvedDebateFacts:
    return ResolvedDebateFacts(
        scoring_mode=scoring_mode,
        equivalence=MappingProxyType(equivalence),
        correctness=MappingProxyType(correctness),
        telemetry=MappingProxyType({}),
    )


# ---------------------------------------------------------------------------
# Layer 1: Tolerant extraction
# ---------------------------------------------------------------------------


class TestTolerantExtract:
    def test_single_positive_in_sentence(self):
        assert _tolerant_extract("I think the answer is SAME.", "SAME", "DIFFERENT") == "SAME"

    def test_single_negative_in_sentence(self):
        assert _tolerant_extract("They are clearly DIFFERENT here.", "SAME", "DIFFERENT") == "DIFFERENT"

    def test_negated_returns_none(self):
        assert _tolerant_extract("They are not SAME at all.", "SAME", "DIFFERENT") is None

    def test_both_present_returns_none(self):
        assert _tolerant_extract("It's SAME but could be DIFFERENT.", "SAME", "DIFFERENT") is None

    def test_no_canonical_returns_none(self):
        assert _tolerant_extract("I think they match.", "SAME", "DIFFERENT") is None

    def test_case_insensitive_match(self):
        assert _tolerant_extract("these two are same", "SAME", "DIFFERENT") == "SAME"

    def test_negation_isnt(self):
        assert _tolerant_extract("it isn't PASS really", "PASS", "FAIL") is None

    def test_negation_neither(self):
        assert _tolerant_extract("neither is PASS", "PASS", "FAIL") is None


# ---------------------------------------------------------------------------
# Layer 2: Error isolation (strict=False)
# ---------------------------------------------------------------------------


class _FailingJudgeClient(AnswerJudgeClient):
    """Client that always returns an unparseable verdict."""

    async def complete_binary(self, *, system: str, user: str, kind: str | None = None) -> str:
        return "absolutely_not_a_verdict_42"


class _InfraFailJudgeClient(AnswerJudgeClient):
    """Client that raises a non-BinaryJudgeError (infra failure)."""

    async def complete_binary(self, *, system: str, user: str, kind: str | None = None) -> str:
        raise ConnectionError("network down")


@pytest.mark.asyncio
async def test_strict_mode_propagates_binary_judge_error():
    """Default strict=True: BinaryJudgeError crashes the batch."""
    state = _state(
        target="water",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="steam", slot_id=1),
        ),
    )
    prompts_ref = "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml"
    state = replace(state, spec=replace(state.spec, prompts_ref=prompts_ref))

    with pytest.raises(BinaryJudgeError):
        await resolve_debate_facts_for_states(
            [state],
            scorer=_FailingJudgeClient(),
            prompts_for_ref=resolve_prompts,
            parallelism=4,
            strict=True,
        )


@pytest.mark.asyncio
async def test_nonstrict_mode_isolates_binary_judge_error():
    """strict=False: BinaryJudgeError is logged, keys are skipped, batch completes."""
    state = _state(
        target="water",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="steam", slot_id=1),
        ),
    )
    prompts_ref = "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml"
    state = replace(state, spec=replace(state.spec, prompts_ref=prompts_ref))

    facts = await resolve_debate_facts_for_states(
        [state],
        scorer=_FailingJudgeClient(),
        prompts_for_ref=resolve_prompts,
        parallelism=4,
        strict=False,
    )

    assert len(facts) == 1
    # Missing keys should be tracked in telemetry.
    tel = facts[0].telemetry
    assert tel["binary_judge_errors.matcher"] >= 0
    assert tel["binary_judge_errors.grader"] >= 0
    # Keys that failed LLM should be missing from the result dicts.
    assert "missing_equivalence_keys" in tel
    assert "missing_correctness_keys" in tel


@pytest.mark.asyncio
async def test_nonstrict_mode_reraises_infra_errors():
    """strict=False: non-BinaryJudgeError (infra) still crashes."""
    state = _state(
        target="water",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="steam", slot_id=1),
        ),
    )
    prompts_ref = "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml"
    state = replace(state, spec=replace(state.spec, prompts_ref=prompts_ref))

    with pytest.raises(ConnectionError):
        await resolve_debate_facts_for_states(
            [state],
            scorer=_InfraFailJudgeClient(),
            prompts_for_ref=resolve_prompts,
            parallelism=4,
            strict=False,
        )


# ---------------------------------------------------------------------------
# Layer 3: Three-valued metric logic
# ---------------------------------------------------------------------------


class TestTruthSurfaced:
    """truth_surfaced: explicit three-valued logic."""

    def test_true_none_gives_1(self):
        """(True, None) -> 1.0: at least one correct."""
        state = _state(
            target="4",
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="4", slot_id=0),
                # B has no answer (None correctness).
            ),
        )
        # Build facts where A is correct, B has no entry (None).
        q = state.spec.problem.task_prompt
        corr_key_a = _grader_key(ScoringMode.MCQ, q, "4", "4")
        facts = _make_facts(ScoringMode.MCQ, {}, {corr_key_a: True})
        metrics = built_in_metric_values(state, facts)
        assert metrics["truth_surfaced"] == 1.0

    def test_none_false_gives_none(self):
        """(None, False) -> None: one unknown, one wrong, can't tell."""
        state = _state(
            target="4",
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                # A has no answer.
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="5", slot_id=1),
            ),
        )
        q = state.spec.problem.task_prompt
        corr_key_b = _grader_key(ScoringMode.MCQ, q, "4", "5")
        facts = _make_facts(ScoringMode.MCQ, {}, {corr_key_b: False})
        metrics = built_in_metric_values(state, facts)
        assert metrics["truth_surfaced"] is None

    def test_false_false_gives_0(self):
        """(False, False) -> 0.0: both wrong."""
        state = _state(
            target="4",
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="5", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="6", slot_id=1),
            ),
        )
        q = state.spec.problem.task_prompt
        corr_key_a = _grader_key(ScoringMode.MCQ, q, "4", "5")
        corr_key_b = _grader_key(ScoringMode.MCQ, q, "4", "6")
        facts = _make_facts(ScoringMode.MCQ, {}, {corr_key_a: False, corr_key_b: False})
        metrics = built_in_metric_values(state, facts)
        assert metrics["truth_surfaced"] == 0.0


class TestStanceChange:
    """stance_change: None lookup -> None (not 0.0)."""

    def test_none_lookup_gives_none(self):
        """With a missing equivalence key, stance_change should be None."""
        state = _state(
            target=None,
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="X", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="Y", slot_id=1),
                _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="Z", slot_id=2),
                _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="Y", slot_id=3),
            ),
        )
        # Build facts with NO equivalence entries -> all lookups return None.
        facts = _make_facts(ScoringMode.MCQ, {}, {})
        metrics = built_in_metric_values(state, facts)
        # A changed X->Z with no equivalence info => None, not 0.0.
        assert metrics["stance_change.debater_a"] is None

    def test_false_lookup_gives_1(self):
        """stance_change with definite difference => 1.0."""
        state = _state(
            target=None,
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="X", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="Y", slot_id=1),
                _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="Z", slot_id=2),
                _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="Y", slot_id=3),
            ),
        )
        q = state.spec.problem.task_prompt
        eq_key = _matcher_key(ScoringMode.MCQ, q, "X", "Z")
        facts = _make_facts(ScoringMode.MCQ, {eq_key: False}, {})
        metrics = built_in_metric_values(state, facts)
        assert metrics["stance_change.debater_a"] == 1.0

    def test_true_lookup_gives_0(self):
        """stance_change with equivalence => 0.0."""
        state = _state(
            target=None,
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="X", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="Y", slot_id=1),
                _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="X", slot_id=2),
                _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="Y", slot_id=3),
            ),
        )
        q = state.spec.problem.task_prompt
        eq_key = _matcher_key(ScoringMode.MCQ, q, "X", "X")
        facts = _make_facts(ScoringMode.MCQ, {eq_key: True}, {})
        metrics = built_in_metric_values(state, facts)
        assert metrics["stance_change.debater_a"] == 0.0


class TestConvergenceRound:
    """convergence_round: unknown earlier round -> None."""

    def test_unknown_earlier_round_blocks_convergence(self):
        """If round 0 equivalence is unknown, round 1 convergence can't be confirmed."""
        state = _state(
            target=None,
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="X", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="Y", slot_id=1),
                _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="Z", slot_id=2),
                _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="Z", slot_id=3),
            ),
        )
        q = state.spec.problem.task_prompt
        # Round 0: X vs Y -> unknown (not in dict).
        # Round 1: Z vs Z -> True.
        eq_key_r1 = _matcher_key(ScoringMode.MCQ, q, "Z", "Z")
        facts = _make_facts(ScoringMode.MCQ, {eq_key_r1: True}, {})
        metrics = built_in_metric_values(state, facts)
        # Can't confirm convergence at round 1 because round 0 is unknown.
        assert metrics["convergence_round"] is None

    def test_false_then_true_gives_convergence(self):
        """Round 0 disagree (False), round 1 agree (True) => convergence at round 1."""
        state = _state(
            target=None,
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="X", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="Y", slot_id=1),
                _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="Z", slot_id=2),
                _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="Z", slot_id=3),
            ),
        )
        q = state.spec.problem.task_prompt
        eq_key_r0 = _matcher_key(ScoringMode.MCQ, q, "X", "Y")
        eq_key_r1 = _matcher_key(ScoringMode.MCQ, q, "Z", "Z")
        facts = _make_facts(ScoringMode.MCQ, {eq_key_r0: False, eq_key_r1: True}, {})
        metrics = built_in_metric_values(state, facts)
        assert metrics["convergence_round"] == 1.0

    def test_true_at_round_0_gives_convergence_0(self):
        """Round 0 agree (True) => convergence at round 0."""
        state = _state(
            target=None,
            scoring_mode=ScoringMode.MCQ,
            transcript=(
                _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="Z", slot_id=0),
                _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="Z", slot_id=1),
            ),
        )
        q = state.spec.problem.task_prompt
        eq_key_r0 = _matcher_key(ScoringMode.MCQ, q, "Z", "Z")
        facts = _make_facts(ScoringMode.MCQ, {eq_key_r0: True}, {})
        metrics = built_in_metric_values(state, facts)
        assert metrics["convergence_round"] == 0.0
