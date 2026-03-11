"""Tests for DebateProblemSpec scoring_mode + DebateDataset homogeneity validation.

With the DebateProblemSpec refactor, scoring_mode is an explicit required field
on each problem. DebateDataset validates that all problems in a batch share
the same scoring_mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.dataset import DebateDataset
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateGameSpec,
    DebateProblemSpec,
    ProtocolKind,
    Role,
    ScoringMode,
)


_GAME = DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=1)


# -- Homogeneous datasets work fine --


def test_mcq_dataset_accepts_homogeneous_problems() -> None:
    problems = [
        DebateProblemSpec.from_seat_answers("Q1", "A", "B", ScoringMode.MCQ),
        DebateProblemSpec.from_seat_answers("Q2", "C", "D", ScoringMode.MCQ),
    ]
    ds = DebateDataset(
        problems=problems,
        batch_size=2,
        group_size=1,
        game=_GAME,
        renderer=MagicMock(),
    )
    assert len(ds.problems) == 2


def test_open_ended_dataset_with_scorer() -> None:
    problems = [
        DebateProblemSpec("Q1", ScoringMode.OPEN_ENDED, target="water"),
        DebateProblemSpec("Q2", ScoringMode.OPEN_ENDED, target="fire"),
    ]
    ds = DebateDataset(
        problems=problems,
        batch_size=2,
        group_size=1,
        game=_GAME,
        renderer=MagicMock(),
        scorer=MagicMock(),
    )
    assert len(ds.problems) == 2


# -- Mixed modes rejected --


def test_mixed_scoring_modes_rejected() -> None:
    problems = [
        DebateProblemSpec.from_seat_answers("Q1", "A", "B", ScoringMode.MCQ),
        DebateProblemSpec("Q2", ScoringMode.OPEN_ENDED, target="water"),
    ]
    with pytest.raises(ValueError, match="same scoring_mode"):
        DebateDataset(
            problems=problems,
            batch_size=2,
            group_size=1,
            game=_GAME,
            renderer=MagicMock(),
        )


# -- Open-ended without scorer rejected --


def test_open_ended_without_scorer_rejected() -> None:
    problems = [
        DebateProblemSpec("Q1", ScoringMode.OPEN_ENDED, target="water"),
    ]
    with pytest.raises(ValueError, match="OPEN_ENDED.*scorer"):
        DebateDataset(
            problems=problems,
            batch_size=1,
            group_size=1,
            game=_GAME,
            renderer=MagicMock(),
        )


# -- DebateProblemSpec normalization --


def test_blank_answers_normalize_to_none() -> None:
    """answer_by_role with all-blank values normalizes to None."""
    p = DebateProblemSpec.from_seat_answers("Q", "", "", ScoringMode.OPEN_ENDED)
    assert p.answer_by_role is None


def test_whitespace_answers_stripped() -> None:
    """Whitespace-only values are stripped to empty and pruned."""
    p = DebateProblemSpec(
        task_prompt="Q",
        scoring_mode=ScoringMode.MCQ,
        answer_by_role={Role.DEBATER_A: "  A  ", Role.DEBATER_B: "  "},
    )
    assert p.answer_by_role is not None
    assert p.answer_by_role[Role.DEBATER_A] == "A"
    assert Role.DEBATER_B not in p.answer_by_role


def test_both_blank_answers_become_none() -> None:
    p = DebateProblemSpec(
        task_prompt="Q",
        scoring_mode=ScoringMode.MCQ,
        answer_by_role={Role.DEBATER_A: "   ", Role.DEBATER_B: ""},
    )
    assert p.answer_by_role is None


# -- from_seat_answers convenience --


def test_from_seat_answers_basic() -> None:
    p = DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ, target="A")
    assert p.task_prompt == "Q"
    assert p.scoring_mode == ScoringMode.MCQ
    assert p.answer_by_role is not None
    assert p.answer_by_role[Role.DEBATER_A] == "A"
    assert p.answer_by_role[Role.DEBATER_B] == "B"
    assert p.target == "A"


def test_from_seat_answers_no_target() -> None:
    p = DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ)
    assert p.target is None


# -- DebateGameSpec validation --


def test_game_spec_num_rounds_validation() -> None:
    with pytest.raises(ValueError, match="num_rounds"):
        DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=0)

    with pytest.raises(ValueError, match="num_rounds"):
        DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=-1)

    # Valid: num_rounds=1
    game = DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=1)
    assert game.num_rounds == 1
