"""Tests for the LLM judge callback and reward functions."""

from __future__ import annotations

import pytest

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import Message

from ..scoring.fields import EnumScoring, FieldSpec
from ..scoring.judge import LLMJudgeCallback, _parse_verdict, zero_sum_outcome_reward, _ENUM_TO_ROLE
from ..core.schedule import build_schedule
from ..types import (
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    JudgeRequest,
    ProtocolKind,
    Role,
    ScoringMode,
    Utterance,
)


# -- Specs for schema-driven tests --

_SELFPLAY_SPECS: dict[str, FieldSpec] = {
    "reasoning": FieldSpec(type=str),
    "decision": FieldSpec(
        type=str,
        scoring=EnumScoring(values=("A", "B", "tie")),
    ),
}

_DEFAULT_SPECS: dict[str, FieldSpec] = {
    "reason": FieldSpec(type=str),
    "decision": FieldSpec(
        type=str,
        scoring=EnumScoring(values=("debater_a", "debater_b", "tie")),
    ),
}


class _MockCompleter(MessageCompleter):
    def __init__(self, response_text: str) -> None:
        self._text = response_text

    async def __call__(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content=self._text)


def _make_request() -> JudgeRequest:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = DebateSpec(
        debate_id="test",
        problem=DebateProblemSpec(
            task_prompt="test question",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
        ),
        schedule=schedule,
        open_reasoning=False,
    )
    transcript = (
        Utterance(
            role=Role.DEBATER_A,
            round_index=0,
            phase=schedule[0].phase,
            text="arg A",
            token_count=2,
            slot_id=schedule[0].slot_id,
        ),
        Utterance(
            role=Role.DEBATER_B,
            round_index=0,
            phase=schedule[1].phase,
            text="arg B",
            token_count=2,
            slot_id=schedule[1].slot_id,
        ),
    )
    state = DebateState(
        spec=spec,
        slot_index=len(schedule),
        rounds_completed=1,
        transcript=transcript,
        pending_simultaneous={},
        judge_trace=(),
        done=True,
        outcome=None,
    )
    return JudgeRequest(state=state, trigger="final")


# -- _ENUM_TO_ROLE mapping --


def test_enum_to_role_covers_all():
    assert _ENUM_TO_ROLE["A"] == Role.DEBATER_A
    assert _ENUM_TO_ROLE["B"] == Role.DEBATER_B
    assert _ENUM_TO_ROLE["debater_a"] == Role.DEBATER_A
    assert _ENUM_TO_ROLE["debater_b"] == Role.DEBATER_B
    assert _ENUM_TO_ROLE["tie"] is None


# -- on_boundary --


@pytest.mark.asyncio
async def test_on_boundary_returns_none() -> None:
    judge = LLMJudgeCallback(_MockCompleter("irrelevant"))
    result = await judge.on_boundary(_make_request())
    assert result is None


# -- _parse_verdict with "A"/"B"/"tie" vocabulary (selfplay packs) --


def test_selfplay_winner_a():
    outcome = _parse_verdict(
        "some reasoning", _SELFPLAY_SPECS, {"reasoning": "good", "decision": "A"}
    )
    assert outcome.winner == Role.DEBATER_A
    assert outcome.scores_by_role[Role.DEBATER_A] == 1.0
    assert outcome.scores_by_role[Role.DEBATER_B] == -1.0


def test_selfplay_winner_b():
    outcome = _parse_verdict(
        "some reasoning", _SELFPLAY_SPECS, {"reasoning": "good", "decision": "B"}
    )
    assert outcome.winner == Role.DEBATER_B
    assert outcome.scores_by_role[Role.DEBATER_B] == 1.0
    assert outcome.scores_by_role[Role.DEBATER_A] == -1.0


def test_selfplay_tie():
    outcome = _parse_verdict(
        "some reasoning", _SELFPLAY_SPECS, {"reasoning": "equal", "decision": "tie"}
    )
    assert outcome.winner is None
    assert outcome.scores_by_role[Role.DEBATER_A] == 0.0
    assert outcome.scores_by_role[Role.DEBATER_B] == 0.0


def test_selfplay_punctuation_stripped():
    """'A.' should classify as 'A' after punctuation stripping."""
    outcome = _parse_verdict("text", _SELFPLAY_SPECS, {"reasoning": "ok", "decision": "A."})
    assert outcome.winner == Role.DEBATER_A


def test_selfplay_case_insensitive():
    """'a' should classify as 'A'."""
    outcome = _parse_verdict("text", _SELFPLAY_SPECS, {"reasoning": "ok", "decision": "a"})
    assert outcome.winner == Role.DEBATER_A


# -- _parse_verdict with "debater_a"/"debater_b"/"tie" vocabulary (default packs) --


def test_default_winner_a():
    outcome = _parse_verdict("text", _DEFAULT_SPECS, {"reason": "good", "decision": "debater_a"})
    assert outcome.winner == Role.DEBATER_A


def test_default_winner_b():
    outcome = _parse_verdict("text", _DEFAULT_SPECS, {"reason": "good", "decision": "debater_b"})
    assert outcome.winner == Role.DEBATER_B


def test_default_case_insensitive():
    outcome = _parse_verdict("text", _DEFAULT_SPECS, {"reason": "ok", "decision": "Debater_A"})
    assert outcome.winner == Role.DEBATER_A


# -- Tie on garbage / missing --


def test_garbage_decision_is_tie():
    outcome = _parse_verdict("text", _SELFPLAY_SPECS, {"reasoning": "ok", "decision": "the judge"})
    assert outcome.winner is None


def test_none_fields_is_tie():
    outcome = _parse_verdict("no xml at all", _SELFPLAY_SPECS, fields=None)
    assert outcome.winner is None
    assert outcome.scores_by_role[Role.DEBATER_A] == 0.0
    assert outcome.scores_by_role[Role.DEBATER_B] == 0.0


def test_missing_decision_field_is_tie():
    outcome = _parse_verdict("text", _SELFPLAY_SPECS, {"reasoning": "I forgot to decide"})
    assert outcome.winner is None


def test_empty_specs_is_tie():
    """No EnumScoring in specs -> tie."""
    bare_specs = {"reason": FieldSpec(type=str)}
    outcome = _parse_verdict("text", bare_specs, {"reason": "ok"})
    assert outcome.winner is None


# -- verdict_text is think-stripped full text --


def test_verdict_text_is_think_stripped():
    text = "<thinking>internal</thinking>The real verdict."
    outcome = _parse_verdict(text, _SELFPLAY_SPECS, {"reasoning": "x", "decision": "A"})
    assert "internal" not in outcome.verdict_text
    assert "The real verdict." in outcome.verdict_text


# -- zero_sum_outcome_reward --


def test_zero_sum_outcome_reward_winner() -> None:
    outcome = DebateOutcome(
        winner=Role.DEBATER_A,
        scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
    )
    rewards = zero_sum_outcome_reward(outcome)
    assert rewards[Role.DEBATER_A] == 1.0
    assert rewards[Role.DEBATER_B] == -1.0


def test_zero_sum_outcome_reward_tie() -> None:
    outcome = DebateOutcome(winner=None, scores_by_role={})
    rewards = zero_sum_outcome_reward(outcome)
    assert rewards[Role.DEBATER_A] == 0.0
    assert rewards[Role.DEBATER_B] == 0.0
