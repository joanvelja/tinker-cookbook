"""Schedule builder for debate protocols."""

from __future__ import annotations

from .types import Phase, ProtocolKind, Role, TurnSlot, VisibilityPolicy


def build_schedule(
    kind: ProtocolKind,
    num_rounds: int,
    *,
    include_judge_turns: bool = False,
) -> tuple[TurnSlot, ...]:
    """Build a turn schedule for the given protocol.

    Returns an ordered tuple of TurnSlot describing every step of the debate.
    """
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

    slots: list[TurnSlot] = []
    slot_id = 0

    for round_idx in range(num_rounds):
        phase = Phase.PROPOSE if round_idx == 0 else Phase.CRITIQUE

        if kind == ProtocolKind.SEQUENTIAL:
            # A then B, sequential. B's slot is the boundary.
            slots.append(TurnSlot(
                slot_id=slot_id, round_index=round_idx, phase=phase,
                actors=(Role.DEBATER_A,), boundary_after=False,
                visibility_policy=VisibilityPolicy.ALL_PRIOR,
            ))
            slot_id += 1
            slots.append(TurnSlot(
                slot_id=slot_id, round_index=round_idx, phase=phase,
                actors=(Role.DEBATER_B,), boundary_after=True,
                visibility_policy=VisibilityPolicy.ALL_PRIOR,
            ))
            slot_id += 1

        elif kind == ProtocolKind.SIMULTANEOUS:
            # A and B simultaneously. Single slot is the boundary.
            slots.append(TurnSlot(
                slot_id=slot_id, round_index=round_idx, phase=phase,
                actors=(Role.DEBATER_A, Role.DEBATER_B), boundary_after=True,
                visibility_policy=VisibilityPolicy.COMPLETED_ROUNDS_ONLY,
            ))
            slot_id += 1

        elif kind == ProtocolKind.HYBRID:
            if round_idx == 0:
                # First round: simultaneous proposals.
                slots.append(TurnSlot(
                    slot_id=slot_id, round_index=round_idx, phase=Phase.PROPOSE,
                    actors=(Role.DEBATER_A, Role.DEBATER_B), boundary_after=True,
                    visibility_policy=VisibilityPolicy.COMPLETED_ROUNDS_ONLY,
                ))
                slot_id += 1
            else:
                # Subsequent rounds: sequential critiques.
                slots.append(TurnSlot(
                    slot_id=slot_id, round_index=round_idx, phase=Phase.CRITIQUE,
                    actors=(Role.DEBATER_A,), boundary_after=False,
                    visibility_policy=VisibilityPolicy.ALL_PRIOR,
                ))
                slot_id += 1
                slots.append(TurnSlot(
                    slot_id=slot_id, round_index=round_idx, phase=Phase.CRITIQUE,
                    actors=(Role.DEBATER_B,), boundary_after=True,
                    visibility_policy=VisibilityPolicy.ALL_PRIOR,
                ))
                slot_id += 1
        else:
            raise ValueError(f"Unknown protocol kind: {kind}")

        # Optionally append judge turns after each boundary.
        if include_judge_turns:
            slots.append(TurnSlot(
                slot_id=slot_id, round_index=round_idx, phase=Phase.JUDGE_QUERY,
                actors=(Role.JUDGE,), boundary_after=False,
                visibility_policy=VisibilityPolicy.ALL_PRIOR,
            ))
            slot_id += 1
            slots.append(TurnSlot(
                slot_id=slot_id, round_index=round_idx, phase=Phase.JUDGE_VERDICT,
                actors=(Role.JUDGE,), boundary_after=False,
                visibility_policy=VisibilityPolicy.ALL_PRIOR,
            ))
            slot_id += 1

    return tuple(slots)
