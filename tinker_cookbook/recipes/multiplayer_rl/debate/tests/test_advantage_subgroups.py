"""Integration tests for DebateGroupBuilder.advantage_subgroups() partitioning."""

from __future__ import annotations

from unittest.mock import MagicMock

from tinker_cookbook.recipes.multiplayer_rl.debate.builders import DebateGroupBuilder
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateGameSpec,
    DebateProblemSpec,
    ProtocolKind,
    Role,
    ScoringMode,
)

_MCQ_PROBLEM = DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ)
_SEQ_1 = DebateGameSpec(protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1)


def _selfplay_builder(
    group_size: int = 1, include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B)
) -> DebateGroupBuilder:
    return DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=MagicMock(),
        group_size=group_size,
        include_roles=include_roles,
    )


def _frozen_opp_builder() -> DebateGroupBuilder:
    return DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=MagicMock(),
        opponent_completer=MagicMock(),
        opponent_renderer=MagicMock(),
        group_size=2,
    )


# --- Self-play mode: subgroups should partition by role ---


def test_selfplay_subgroups_two_roles_group_size_1():
    builder = _selfplay_builder(group_size=1)
    subgroups = builder.advantage_subgroups(n_trajectories=2)
    assert subgroups == ((0,), (1,))


def test_selfplay_subgroups_two_roles_group_size_3():
    builder = _selfplay_builder(group_size=3)
    subgroups = builder.advantage_subgroups(n_trajectories=6)
    assert subgroups == ((0, 2, 4), (1, 3, 5))


def test_selfplay_subgroups_match_interleaving():
    """Subgroup indices match the actual env layout: role A at even, role B at odd."""
    builder = _selfplay_builder(group_size=4)
    subgroups = builder.advantage_subgroups(n_trajectories=8)
    assert subgroups[0] == (0, 2, 4, 6)
    assert subgroups[1] == (1, 3, 5, 7)


# --- Frozen-opponent mode: no subgroups ---


def test_frozen_opponent_returns_none():
    builder = _frozen_opp_builder()
    subgroups = builder.advantage_subgroups(n_trajectories=2)
    assert subgroups is None


# --- Edge cases ---


def test_single_trajectory():
    builder = _selfplay_builder(group_size=1, include_roles=(Role.DEBATER_A, Role.DEBATER_B))
    subgroups = builder.advantage_subgroups(n_trajectories=1)
    assert len(subgroups) == 2
    assert subgroups[0] == (0,)
    assert subgroups[1] == ()  # no index 1 exists
