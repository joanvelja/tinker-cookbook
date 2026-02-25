"""Transcript query helpers for debate state."""
from __future__ import annotations

from ..types import DebateState, Phase, Role, Utterance

Answer = str


def answer_from_utterance(utterance: Utterance) -> Answer | None:
    """Return utterance.fields['answer'] if present, else None."""
    if utterance.fields is None:
        return None
    return utterance.fields.get("answer")


def filter_utterances(
    state: DebateState, *, role: Role | None = None, phase: Phase | None = None
) -> tuple[Utterance, ...]:
    """Transcript-order filter by role/phase."""
    result = []
    for u in state.transcript:
        if role is not None and u.role != role:
            continue
        if phase is not None and u.phase != phase:
            continue
        result.append(u)
    return tuple(result)


def answer_at_round(
    state: DebateState, *, role: Role, round_index: int, phase: Phase | None = None
) -> Answer | None:
    """Answer from the last utterance by role in that round."""
    for u in reversed(state.transcript):
        if u.role == role and u.round_index == round_index:
            if phase is not None and u.phase != phase:
                continue
            ans = answer_from_utterance(u)
            if ans is not None:
                return ans
    return None


def final_answer(
    state: DebateState, *, role: Role, phase: Phase | None = None
) -> Answer | None:
    """Answer from the last utterance by role in the transcript."""
    for u in reversed(state.transcript):
        if u.role == role:
            if phase is not None and u.phase != phase:
                continue
            ans = answer_from_utterance(u)
            if ans is not None:
                return ans
    return None


def answers_by_round(
    state: DebateState, *, role: Role, phase: Phase | None = None
) -> list[Answer | None]:
    """List indexed by round_index. Latest answer per round (critique overrides propose)."""
    schedule = state.spec.schedule
    num_rounds = max((s.round_index for s in schedule), default=0) + 1
    result: list[Answer | None] = [None] * num_rounds
    for u in state.transcript:
        if u.role != role:
            continue
        if phase is not None and u.phase != phase:
            continue
        ans = answer_from_utterance(u)
        if ans is not None and u.round_index < num_rounds:
            result[u.round_index] = ans
    return result
