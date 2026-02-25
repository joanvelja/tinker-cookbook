"""Pure state transitions for the debate environment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .types import (
    ActionResult,
    DebateState,
    JudgeDecision,
    Role,
    TurnSlot,
    Utterance,
)


def get_current_slot(state: DebateState) -> TurnSlot | None:
    """Current slot or None if schedule exhausted."""
    schedule = state.spec.schedule
    if state.slot_index < len(schedule):
        return schedule[state.slot_index]
    return None


def get_eligible_roles(state: DebateState) -> frozenset[Role]:
    """Roles that can act now (current slot actors minus already-pending)."""
    slot = get_current_slot(state)
    if slot is None or state.done:
        return frozenset()
    return frozenset(slot.actors) - frozenset(state.pending_simultaneous)


def _advance(state: DebateState, slot: TurnSlot, committed: tuple[Utterance, ...]) -> ActionResult:
    """Advance slot_index, handle boundary and schedule exhaustion."""
    new_slot_index = state.slot_index + 1
    new_rounds_completed = state.rounds_completed + (1 if slot.boundary_after else 0)
    schedule_exhausted = new_slot_index >= len(state.spec.schedule)
    new_state = replace(
        state,
        slot_index=new_slot_index,
        rounds_completed=new_rounds_completed,
        transcript=state.transcript + committed,
        pending_simultaneous={},
        done=schedule_exhausted,
    )
    return ActionResult(
        new_state=new_state,
        committed=committed,
        boundary_reached=slot.boundary_after,
        episode_done=schedule_exhausted,
    )


def apply_action(state: DebateState, role: Role, text: str, token_count: int, fields: Mapping[str, Any] | None = None) -> ActionResult:
    """Apply a single action. For sequential slots, commits immediately.
    For simultaneous slots, buffers until all actors have submitted."""
    slot = get_current_slot(state)
    if slot is None:
        raise ValueError("Schedule exhausted — no current slot.")
    if role not in get_eligible_roles(state):
        raise ValueError(
            f"{role} is not eligible. Eligible: {get_eligible_roles(state)}"
        )

    utterance = Utterance(
        role=role,
        round_index=slot.round_index,
        phase=slot.phase,
        text=text,
        token_count=token_count,
        slot_id=slot.slot_id,
        fields=fields,
    )

    is_simultaneous = len(slot.actors) > 1

    if not is_simultaneous:
        # Sequential: commit immediately and advance.
        return _advance(state, slot, (utterance,))

    # Simultaneous: buffer in pending_simultaneous.
    new_pending = dict(state.pending_simultaneous)
    new_pending[role] = utterance
    all_present = frozenset(new_pending) == frozenset(slot.actors)

    if not all_present:
        # Still waiting for other actors.
        return ActionResult(
            new_state=replace(state, pending_simultaneous=new_pending),
            committed=(),
            boundary_reached=False,
            episode_done=False,
        )

    # All actors present — commit in canonical order.
    committed = tuple(new_pending[r] for r in slot.actors)
    state_with_pending = replace(state, pending_simultaneous=new_pending)
    return _advance(state_with_pending, slot, committed)


def commit_slot_actions(
    state: DebateState, actions_by_role: Mapping[Role, tuple[str, int] | tuple[str, int, Mapping[str, Any] | None]]
) -> ActionResult:
    """Simultaneous slot: atomic commit in canonical order from TurnSlot.actors tuple."""
    slot = get_current_slot(state)
    if slot is None:
        raise ValueError("Schedule exhausted — no current slot.")
    if frozenset(actions_by_role) != frozenset(slot.actors):
        raise ValueError(
            f"Expected actions from {set(slot.actors)}, got {set(actions_by_role)}"
        )

    committed = tuple(
        Utterance(
            role=r,
            round_index=slot.round_index,
            phase=slot.phase,
            text=actions_by_role[r][0],
            token_count=actions_by_role[r][1],
            slot_id=slot.slot_id,
            fields=actions_by_role[r][2] if len(actions_by_role[r]) > 2 else None,
        )
        for r in slot.actors
    )
    return _advance(state, slot, committed)


def apply_judge_event(state: DebateState, decision: JudgeDecision) -> DebateState:
    """Append to judge_trace."""
    return replace(
        state,
        judge_trace=state.judge_trace + (decision,),
    )


def fork_state(state: DebateState) -> DebateState:
    """Return state as-is (already frozen/immutable)."""
    return state
