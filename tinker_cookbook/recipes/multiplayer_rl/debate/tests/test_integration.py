"""Integration tests for the scoring pipeline.

Tests the full data path: prompts -> field extraction -> reducer -> trajectory -> metrics.
No Tinker API needed -- all tests run offline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    Utterance,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.core.reducer import apply_action
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.trajectory import (
    answer_from_utterance,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.metrics import (
    accuracy,
    concession_correctness,
    debater_accuracy_delta,
    disagreement,
    draw_rate,
    judge_quality,
    stance_change,
    truth_surfaced,
    truth_win_if_disagreement,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.parsing import extract_fields
from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.judge import _parse_verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    target: str | None = "C",
    num_rounds: int = 2,
    prompts_ref: str = "scientific_mcq",
) -> DebateSpec:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, num_rounds)
    return DebateSpec(
        debate_id="test",
        task_prompt="Which letter? A) X B) Y C) Z D) W",
        answer_by_role={Role.DEBATER_A: "C", Role.DEBATER_B: "B"},
        schedule=schedule,
        open_reasoning=False,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref=prompts_ref,
        target=target,
    )


def _empty_state(spec: DebateSpec) -> DebateState:
    return DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None,
    )


def _utt(
    role: Role,
    round_index: int,
    phase: Phase,
    slot_id: int,
    text: str,
    answer: str | None = None,
    **extra_fields: object,
) -> Utterance:
    fields = None
    if answer is not None:
        fields = {"answer": answer, **extra_fields}
    return Utterance(
        role=role,
        round_index=round_index,
        phase=phase,
        text=text,
        token_count=len(text.split()),
        slot_id=slot_id,
        fields=fields,
    )


def _build_full_state(
    *,
    target: str | None = "C",
    a_answers: tuple[str, str] = ("C", "C"),
    b_answers: tuple[str, str] = ("B", "C"),
    winner: Role | None = Role.DEBATER_B,
    include_outcome: bool = True,
) -> DebateState:
    """Build a completed 2-round sequential debate state."""
    spec = _make_spec(target=target)
    transcript = (
        _utt(
            Role.DEBATER_A, 0, Phase.PROPOSE, 0, "A proposes", answer=a_answers[0], reasoning="test"
        ),
        _utt(
            Role.DEBATER_B, 0, Phase.PROPOSE, 1, "B proposes", answer=b_answers[0], reasoning="test"
        ),
        _utt(
            Role.DEBATER_A,
            1,
            Phase.CRITIQUE,
            2,
            "A critiques",
            answer=a_answers[1],
            rebuttal="test",
        ),
        _utt(
            Role.DEBATER_B,
            1,
            Phase.CRITIQUE,
            3,
            "B critiques",
            answer=b_answers[1],
            rebuttal="test",
        ),
    )
    outcome = None
    if include_outcome:
        if winner is not None:
            loser = Role.DEBATER_B if winner == Role.DEBATER_A else Role.DEBATER_A
            scores = {winner: 1.0, loser: -1.0}
        else:
            scores = {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
        outcome = DebateOutcome(winner=winner, scores_by_role=scores, verdict_text="test verdict")

    return DebateState(
        spec=spec,
        slot_index=len(spec.schedule),
        rounds_completed=2,
        transcript=transcript,
        pending_simultaneous={},
        judge_trace=(),
        done=True,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Test 1: Field extraction roundtrip
# ---------------------------------------------------------------------------


class TestFieldExtractionRoundtrip:
    def test_extract_answer_and_reasoning(self):
        prompts = resolve_prompts("scientific_mcq")
        specs = prompts.get_field_specs("debater_a", "propose")
        assert specs is not None

        fields = extract_fields("<answer>C</answer><reasoning>because pi</reasoning>", specs)
        assert fields is not None
        assert fields["answer"] == "C"
        assert fields["reasoning"] == "because pi"

    def test_answer_normalized_to_uppercase(self):
        prompts = resolve_prompts("scientific_mcq")
        specs = prompts.get_field_specs("debater_a", "propose")
        assert specs is not None

        fields = extract_fields("<answer>c</answer><reasoning>test</reasoning>", specs)
        assert fields is not None
        # Enum normalizer should capitalize
        assert fields["answer"] == "C"


# ---------------------------------------------------------------------------
# Test 2: Utterance.fields through reducer
# ---------------------------------------------------------------------------


class TestUtteranceFieldsThroughReducer:
    def test_fields_populated_after_apply_action(self):
        spec = _make_spec()
        state = _empty_state(spec)

        result = apply_action(
            state,
            Role.DEBATER_A,
            "<answer>C</answer><reasoning>test</reasoning>",
            token_count=10,
            fields={"answer": "C", "reasoning": "test"},
        )
        assert len(result.committed) == 1
        utt = result.committed[0]
        assert utt.fields is not None
        assert utt.fields["answer"] == "C"
        assert answer_from_utterance(utt) == "C"


# ---------------------------------------------------------------------------
# Test 3: Full 2-round debate metrics
# ---------------------------------------------------------------------------


class TestFull2RoundDebateMetrics:
    """A: C->C (consistent, correct), B: B->C (changed to correct), judge picks B."""

    @pytest.fixture()
    def state(self) -> DebateState:
        return _build_full_state(
            target="C",
            a_answers=("C", "C"),
            b_answers=("B", "C"),
            winner=Role.DEBATER_B,
        )

    def test_accuracy_debater_a(self, state: DebateState):
        assert accuracy(Role.DEBATER_A)(state).value == 1.0

    def test_accuracy_debater_b(self, state: DebateState):
        assert accuracy(Role.DEBATER_B)(state).value == 1.0

    def test_judge_quality(self, state: DebateState):
        # Judge picked B, B is correct -> 1.0
        assert judge_quality()(state).value == 1.0

    def test_truth_surfaced(self, state: DebateState):
        assert truth_surfaced()(state).value == 1.0

    def test_stance_change_a(self, state: DebateState):
        # A: C->C, no change
        assert stance_change(Role.DEBATER_A)(state).value == 0.0

    def test_stance_change_b(self, state: DebateState):
        # B: B->C, changed
        assert stance_change(Role.DEBATER_B)(state).value == 1.0

    def test_concession_correctness_b(self, state: DebateState):
        # B went from wrong (B) to correct (C) -> +1.0
        assert concession_correctness(Role.DEBATER_B)(state).value == 1.0

    def test_debater_accuracy_delta_b(self, state: DebateState):
        # B: wrong -> correct -> +1.0
        assert debater_accuracy_delta(Role.DEBATER_B)(state).value == 1.0

    def test_disagreement(self, state: DebateState):
        # Final answers both C -> no disagreement
        assert disagreement()(state).value == 0.0


# ---------------------------------------------------------------------------
# Test 4: Disagreement + truth_win
# ---------------------------------------------------------------------------


class TestDisagreementTruthWin:
    """A=C (correct), B=B (wrong). Test truth_win under different judge picks."""

    def _state_with_winner(self, winner: Role) -> DebateState:
        return _build_full_state(
            target="C",
            a_answers=("C", "C"),
            b_answers=("B", "B"),
            winner=winner,
        )

    def test_judge_picks_correct(self):
        state = self._state_with_winner(Role.DEBATER_A)
        assert truth_win_if_disagreement()(state).value == 1.0

    def test_judge_picks_wrong(self):
        state = self._state_with_winner(Role.DEBATER_B)
        assert truth_win_if_disagreement()(state).value == 0.0


# ---------------------------------------------------------------------------
# Test 5: No target
# ---------------------------------------------------------------------------


class TestNoTarget:
    @pytest.fixture()
    def state(self) -> DebateState:
        return _build_full_state(target=None)

    def test_accuracy_none(self, state: DebateState):
        assert accuracy(Role.DEBATER_A)(state).value is None

    def test_judge_quality_none(self, state: DebateState):
        assert judge_quality()(state).value is None

    def test_truth_win_none(self, state: DebateState):
        assert truth_win_if_disagreement()(state).value is None

    def test_truth_surfaced_none(self, state: DebateState):
        assert truth_surfaced()(state).value is None

    def test_concession_correctness_none(self, state: DebateState):
        assert concession_correctness(Role.DEBATER_A)(state).value is None


# ---------------------------------------------------------------------------
# Test 6: No outcome
# ---------------------------------------------------------------------------


class TestNoOutcome:
    @pytest.fixture()
    def state(self) -> DebateState:
        return _build_full_state(include_outcome=False)

    def test_judge_quality_none(self, state: DebateState):
        assert judge_quality()(state).value is None

    def test_truth_win_none(self, state: DebateState):
        assert truth_win_if_disagreement()(state).value is None

    def test_draw_rate_none(self, state: DebateState):
        assert draw_rate()(state).value is None


# ---------------------------------------------------------------------------
# Test 7: Judge verdict parsing roundtrip
# ---------------------------------------------------------------------------


class TestJudgeVerdictParsing:
    def test_schema_debater_a(self):
        outcome = _parse_verdict("", fields={"decision": "debater_a"})
        assert outcome.winner == Role.DEBATER_A

    def test_schema_tie(self):
        outcome = _parse_verdict("", fields={"decision": "tie"})
        assert outcome.winner is None

    def test_regex_fallback(self):
        outcome = _parse_verdict("<decision>debater_b</decision>", fields=None)
        assert outcome.winner == Role.DEBATER_B

    def test_garbage_decision(self):
        outcome = _parse_verdict("", fields={"decision": "idk"})
        assert outcome.winner is None


# ---------------------------------------------------------------------------
# Test 8: dump_io with scientific_mcq
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 8b: check_ab_symmetry covers fields
# ---------------------------------------------------------------------------


class TestFieldSymmetryCheck:
    def test_symmetric_fields_no_warnings(self):
        prompts = resolve_prompts("scientific_mcq")
        from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import check_ab_symmetry

        warnings = check_ab_symmetry(prompts)
        assert not any("fields" in w for w in warnings)

    def test_asymmetric_field_triggers_detected(self):
        """Synthetic test: asymmetric triggers should produce a warning."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import (
            check_ab_symmetry,
            DebatePrompts,
        )

        prompts = resolve_prompts("scientific_mcq")
        # Patch fields to create asymmetry (debater_b missing 'critique' trigger).
        asymmetric_fields = {
            "debater_a": prompts.fields.get("debater_a", {}),
            "debater_b": {"propose": prompts.fields.get("debater_b", {}).get("propose", {})},
        }
        patched = DebatePrompts(
            system=prompts.system,
            user=prompts.user,
            question=prompts.question,
            think=prompts.think,
            prefill=prompts.prefill,
            fields=asymmetric_fields,
            content_hash="test",
            source_ref="test",
        )
        warnings = check_ab_symmetry(patched)
        assert any("fields" in w and "trigger" in w for w in warnings)

    def test_asymmetric_field_names_detected(self):
        """Synthetic test: same triggers but different field names should warn."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import (
            check_ab_symmetry,
            DebatePrompts,
        )
        from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.fields import FieldSpec

        prompts = resolve_prompts("scientific_mcq")
        # debater_b.propose has extra field
        b_propose = dict(prompts.fields.get("debater_b", {}).get("propose", {}))
        b_propose["extra_field"] = FieldSpec(str)
        patched_fields = {
            "debater_a": prompts.fields.get("debater_a", {}),
            "debater_b": {**prompts.fields.get("debater_b", {}), "propose": b_propose},
        }
        patched = DebatePrompts(
            system=prompts.system,
            user=prompts.user,
            question=prompts.question,
            think=prompts.think,
            prefill=prompts.prefill,
            fields=patched_fields,
            content_hash="test",
            source_ref="test",
        )
        warnings = check_ab_symmetry(patched)
        assert any("fields.propose" in w for w in warnings)


# ---------------------------------------------------------------------------
# Test 8c: frozen-opponent rejects include_judge_turns
# ---------------------------------------------------------------------------


class TestFrozenOpponentJudgeValidation:
    def test_include_judge_turns_raises(self):
        from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateGroupBuilder

        async def _run():
            builder = DebateGroupBuilder(
                task_prompt="Q?",
                answer_a="A",
                answer_b="B",
                renderer=None,  # type: ignore[arg-type]  # never reached
                protocol_kind=ProtocolKind.SEQUENTIAL,
                num_rounds=1,
                include_judge_turns=True,
                opponent_completer=lambda msgs: {"content": "test"},
                prompts_ref="scientific_mcq",
            )
            await builder.make_envs()

        import asyncio

        with pytest.raises(ValueError, match="include_judge_turns"):
            asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Test 8d: runtime phase-to-trigger mapping
# ---------------------------------------------------------------------------


class TestPhaseToTriggerMapping:
    def test_judge_verdict_maps_to_final(self):
        from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import _PHASE_TO_TRIGGER

        assert _PHASE_TO_TRIGGER["judge_verdict"] == "final"
        assert _PHASE_TO_TRIGGER["judge_query"] == "boundary"

    def test_debater_phases_pass_through(self):
        from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import _PHASE_TO_TRIGGER

        # Debater phases should not be in the mapping (pass through as-is)
        assert "propose" not in _PHASE_TO_TRIGGER
        assert "critique" not in _PHASE_TO_TRIGGER


# ---------------------------------------------------------------------------
# Test 9: dump_io with scientific_mcq
# ---------------------------------------------------------------------------


class TestDumpIO:
    def test_dump_io_scientific_mcq(self):
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io",
                "--prompts",
                "scientific_mcq",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0, f"dump_io failed:\n{result.stderr}"
        assert "A, B, C, D" in result.stdout  # field instructions visible
