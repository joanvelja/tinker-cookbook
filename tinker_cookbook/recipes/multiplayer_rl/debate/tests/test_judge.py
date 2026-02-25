"""Tests for the LLM judge callback and reward functions."""

from __future__ import annotations

import pytest

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import Message

from ..scoring.judge import LLMJudgeCallback, _extract_xml_fields, _parse_verdict, zero_sum_outcome_reward
from ..core.schedule import build_schedule
from ..types import (
    DebateOutcome,
    DebateSpec,
    DebateState,
    JudgeRequest,
    ProtocolKind,
    Role,
    Utterance,
)


class _MockCompleter(MessageCompleter):
    def __init__(self, response_text: str) -> None:
        self._text = response_text

    async def __call__(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content=self._text)


def _make_request() -> JudgeRequest:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = DebateSpec(
        debate_id="test",
        task_prompt="test question",
        answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
        schedule=schedule,
        open_reasoning=False,
    )
    transcript = (
        Utterance(role=Role.DEBATER_A, round_index=0, phase=schedule[0].phase, text="arg A", token_count=2, slot_id=schedule[0].slot_id),
        Utterance(role=Role.DEBATER_B, round_index=0, phase=schedule[1].phase, text="arg B", token_count=2, slot_id=schedule[1].slot_id),
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


# -- XML extraction --


def test_extract_xml_fields_basic():
    fields = _extract_xml_fields("<decision>debater_a</decision><reason>good args</reason>")
    assert fields == {"decision": "debater_a", "reason": "good args"}


def test_extract_xml_fields_with_prose():
    text = "I think that <decision>debater_b</decision> won because <reason>stronger evidence</reason>."
    fields = _extract_xml_fields(text)
    assert fields["decision"] == "debater_b"
    assert fields["reason"] == "stronger evidence"


def test_extract_xml_fields_multiline():
    text = "<reason>\nline 1\nline 2\n</reason>"
    fields = _extract_xml_fields(text)
    assert fields["reason"] == "line 1\nline 2"


def test_extract_xml_fields_empty():
    assert _extract_xml_fields("no tags here") == {}


# -- on_boundary --


@pytest.mark.asyncio
async def test_on_boundary_returns_none() -> None:
    judge = LLMJudgeCallback(_MockCompleter("irrelevant"))
    result = await judge.on_boundary(_make_request())
    assert result is None


# -- on_final with valid XML --


@pytest.mark.asyncio
async def test_valid_winner_a() -> None:
    text = "<decision>debater_a</decision><reason>better arguments</reason>"
    judge = LLMJudgeCallback(_MockCompleter(text))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner == Role.DEBATER_A
    assert outcome.scores_by_role[Role.DEBATER_A] == 1.0
    assert outcome.scores_by_role[Role.DEBATER_B] == -1.0
    assert outcome.verdict_text == "better arguments"


@pytest.mark.asyncio
async def test_valid_winner_b() -> None:
    text = "<decision>debater_b</decision><reason>more convincing</reason>"
    judge = LLMJudgeCallback(_MockCompleter(text))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner == Role.DEBATER_B
    assert outcome.scores_by_role[Role.DEBATER_B] == 1.0
    assert outcome.scores_by_role[Role.DEBATER_A] == -1.0
    assert outcome.verdict_text == "more convincing"


@pytest.mark.asyncio
async def test_valid_winner_with_preamble() -> None:
    text = "Let me think... <decision>debater_a</decision> <reason>stronger evidence</reason> That's my verdict."
    judge = LLMJudgeCallback(_MockCompleter(text))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner == Role.DEBATER_A
    assert outcome.verdict_text == "stronger evidence"


@pytest.mark.asyncio
async def test_case_insensitive_decision() -> None:
    text = "<decision>Debater_A</decision><reason>won</reason>"
    judge = LLMJudgeCallback(_MockCompleter(text))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner == Role.DEBATER_A


@pytest.mark.asyncio
async def test_tie_decision() -> None:
    text = "<decision>tie</decision><reason>equal</reason>"
    judge = LLMJudgeCallback(_MockCompleter(text))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner is None
    assert outcome.scores_by_role[Role.DEBATER_A] == 0.0
    assert outcome.scores_by_role[Role.DEBATER_B] == 0.0
    assert outcome.verdict_text == "equal"


# -- on_final with garbage --


@pytest.mark.asyncio
async def test_garbage_response_is_tie() -> None:
    judge = LLMJudgeCallback(_MockCompleter("no xml at all"))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner is None
    assert outcome.scores_by_role[Role.DEBATER_A] == 0.0
    assert outcome.scores_by_role[Role.DEBATER_B] == 0.0
    assert outcome.verdict_text == "no xml at all"


@pytest.mark.asyncio
async def test_missing_decision_tag_is_tie() -> None:
    judge = LLMJudgeCallback(_MockCompleter("<reason>I forgot to decide</reason>"))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner is None


@pytest.mark.asyncio
async def test_invalid_decision_value_is_tie() -> None:
    judge = LLMJudgeCallback(_MockCompleter("<decision>the judge</decision>"))
    outcome = await judge.on_final(_make_request())
    assert outcome.winner is None


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
