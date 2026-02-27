"""Tests for the debate reducer."""

from __future__ import annotations

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.core.reducer import (
    apply_action,
    apply_judge_event,
    commit_slot_actions,
    fork_state,
    get_current_slot,
    get_eligible_roles,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateSpec,
    DebateState,
    JudgeDecision,
    ProtocolKind,
    Role,
)


def _make_state(
    kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    num_rounds: int = 2,
) -> DebateState:
    schedule = build_schedule(kind, num_rounds)
    spec = DebateSpec(
        debate_id="test",
        task_prompt="Which is bigger?",
        answer_by_role={Role.DEBATER_A: "2", Role.DEBATER_B: "3"},
        schedule=schedule,
        open_reasoning=False,
    )
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


# --- get_current_slot ---


def test_get_current_slot():
    state = _make_state()
    slot = get_current_slot(state)
    assert slot is not None
    assert slot.slot_id == 0
    assert slot.actors == (Role.DEBATER_A,)


def test_get_current_slot_exhausted():
    state = _make_state(num_rounds=1)
    from dataclasses import replace

    exhausted = replace(state, slot_index=len(state.spec.schedule))
    assert get_current_slot(exhausted) is None


# --- get_eligible_roles ---


def test_eligible_roles_sequential():
    state = _make_state()
    assert get_eligible_roles(state) == frozenset({Role.DEBATER_A})


def test_eligible_roles_simultaneous():
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
    assert get_eligible_roles(state) == frozenset({Role.DEBATER_A, Role.DEBATER_B})


def test_eligible_roles_done():
    from dataclasses import replace

    state = replace(_make_state(), done=True)
    assert get_eligible_roles(state) == frozenset()


# --- Sequential full playthrough (2 rounds) ---


def test_sequential_full_playthrough():
    state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=2)
    # Round 0: A proposes
    result = apply_action(state, Role.DEBATER_A, "A proposes", 2)
    assert len(result.committed) == 1
    assert not result.boundary_reached
    assert not result.episode_done
    state = result.new_state
    assert state.slot_index == 1
    assert state.rounds_completed == 0

    # Round 0: B proposes (boundary)
    result = apply_action(state, Role.DEBATER_B, "B proposes", 2)
    assert result.boundary_reached
    assert not result.episode_done
    state = result.new_state
    assert state.slot_index == 2
    assert state.rounds_completed == 1

    # Round 1: A critiques
    result = apply_action(state, Role.DEBATER_A, "A critiques", 2)
    assert not result.boundary_reached
    state = result.new_state

    # Round 1: B critiques (boundary + done)
    result = apply_action(state, Role.DEBATER_B, "B critiques", 2)
    assert result.boundary_reached
    assert result.episode_done
    state = result.new_state
    assert state.done
    assert state.rounds_completed == 2
    assert len(state.transcript) == 4


# --- Simultaneous full playthrough (2 rounds) ---


def test_simultaneous_full_playthrough():
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=2)

    # Round 0: A submits (buffers)
    result = apply_action(state, Role.DEBATER_A, "A proposes", 2)
    assert len(result.committed) == 0
    assert not result.boundary_reached
    state = result.new_state
    assert len(state.pending_simultaneous) == 1
    assert state.slot_index == 0  # hasn't advanced

    # Round 0: B submits (commits both)
    result = apply_action(state, Role.DEBATER_B, "B proposes", 2)
    assert len(result.committed) == 2
    assert result.boundary_reached
    state = result.new_state
    assert state.slot_index == 1
    assert state.rounds_completed == 1
    assert len(state.pending_simultaneous) == 0

    # Canonical order: A before B
    assert result.committed[0].role == Role.DEBATER_A
    assert result.committed[1].role == Role.DEBATER_B

    # Round 1: B submits first this time (buffers)
    result = apply_action(state, Role.DEBATER_B, "B critiques", 2)
    assert len(result.committed) == 0
    state = result.new_state

    # Round 1: A submits (commits both, episode done)
    result = apply_action(state, Role.DEBATER_A, "A critiques", 2)
    assert len(result.committed) == 2
    assert result.boundary_reached
    assert result.episode_done
    state = result.new_state
    assert state.done
    assert len(state.transcript) == 4

    # Canonical order preserved even though B submitted first
    assert result.committed[0].role == Role.DEBATER_A
    assert result.committed[1].role == Role.DEBATER_B


# --- Hybrid full playthrough (2 rounds) ---


def test_hybrid_full_playthrough():
    state = _make_state(kind=ProtocolKind.HYBRID, num_rounds=2)

    # Round 0: simultaneous proposals
    result = apply_action(state, Role.DEBATER_A, "A proposes", 2)
    assert len(result.committed) == 0
    state = result.new_state
    result = apply_action(state, Role.DEBATER_B, "B proposes", 2)
    assert len(result.committed) == 2
    assert result.boundary_reached
    state = result.new_state
    assert state.rounds_completed == 1

    # Round 1: sequential critiques
    result = apply_action(state, Role.DEBATER_A, "A critiques", 2)
    assert not result.boundary_reached
    state = result.new_state
    result = apply_action(state, Role.DEBATER_B, "B critiques", 2)
    assert result.boundary_reached
    assert result.episode_done
    state = result.new_state
    assert state.done
    assert len(state.transcript) == 4


# --- Wrong role rejection ---


def test_wrong_role_rejected():
    state = _make_state(kind=ProtocolKind.SEQUENTIAL)
    # Slot 0 expects A only
    with pytest.raises(ValueError, match="not eligible"):
        apply_action(state, Role.DEBATER_B, "wrong turn", 2)


def test_simultaneous_duplicate_rejected():
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
    result = apply_action(state, Role.DEBATER_A, "A first", 2)
    state = result.new_state
    # A already pending
    with pytest.raises(ValueError, match="not eligible"):
        apply_action(state, Role.DEBATER_A, "A again", 2)


# --- commit_slot_actions ---


def test_commit_slot_actions():
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
    result = commit_slot_actions(
        state,
        {Role.DEBATER_A: ("A says", 2), Role.DEBATER_B: ("B says", 2)},
    )
    assert len(result.committed) == 2
    assert result.committed[0].role == Role.DEBATER_A
    assert result.committed[1].role == Role.DEBATER_B
    assert result.boundary_reached


def test_commit_slot_actions_wrong_roles():
    state = _make_state(kind=ProtocolKind.SEQUENTIAL)
    with pytest.raises(ValueError, match="Expected actions"):
        commit_slot_actions(
            state,
            {Role.DEBATER_A: ("A says", 2), Role.DEBATER_B: ("B says", 2)},
        )


# --- apply_judge_event ---


def test_apply_judge_event():
    state = _make_state()
    decision = JudgeDecision(
        round_index=0,
        verdict="A is better",
        score_delta_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
    )
    new_state = apply_judge_event(state, decision)
    assert len(new_state.judge_trace) == 1
    assert new_state.judge_trace[0] is decision


# --- fork_state ---


def test_fork_state_returns_same_frozen_state():
    state = _make_state()
    forked = fork_state(state)
    assert forked is state  # already frozen, no copy needed


# --- Version monotonicity ---


def test_version_increases():
    state = _make_state(kind=ProtocolKind.SEQUENTIAL)
    v0 = state.version
    result = apply_action(state, Role.DEBATER_A, "hello", 1)
    v1 = result.new_state.version
    assert v1 > v0


def test_version_increases_during_simultaneous_buffer():
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
    v0 = state.version
    result = apply_action(state, Role.DEBATER_A, "A says", 2)
    v1 = result.new_state.version
    assert v1 > v0  # pending adds to version


# --- Fields threading ---


def test_apply_action_with_fields():
    """apply_action propagates fields to the committed Utterance."""
    state = _make_state(kind=ProtocolKind.SEQUENTIAL)
    fields = {"answer": "C", "confidence": 0.9}
    result = apply_action(state, Role.DEBATER_A, "I think C", 3, fields=fields)
    assert len(result.committed) == 1
    utt = result.committed[0]
    assert utt.fields is not None
    assert utt.fields["answer"] == "C"
    assert utt.fields["confidence"] == 0.9


def test_apply_action_without_fields():
    """apply_action with no fields (backward compat) leaves Utterance.fields as None."""
    state = _make_state(kind=ProtocolKind.SEQUENTIAL)
    result = apply_action(state, Role.DEBATER_A, "I think C", 3)
    assert len(result.committed) == 1
    assert result.committed[0].fields is None


def test_commit_slot_actions_with_fields():
    """commit_slot_actions threads fields through 3-tuple entries."""
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
    result = commit_slot_actions(
        state,
        {
            Role.DEBATER_A: ("A says", 2, {"answer": "X"}),
            Role.DEBATER_B: ("B says", 2, {"answer": "Y"}),
        },
    )
    assert len(result.committed) == 2
    assert result.committed[0].fields is not None
    assert result.committed[0].fields["answer"] == "X"
    assert result.committed[1].fields["answer"] == "Y"


def test_commit_slot_actions_without_fields():
    """commit_slot_actions with 2-tuple entries (backward compat) leaves fields None."""
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
    result = commit_slot_actions(
        state,
        {Role.DEBATER_A: ("A says", 2), Role.DEBATER_B: ("B says", 2)},
    )
    assert result.committed[0].fields is None
    assert result.committed[1].fields is None
