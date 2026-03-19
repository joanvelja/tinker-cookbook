"""Smoke tests for experiment TODOs: format penalty and inter-epoch shuffle."""

from __future__ import annotations

from unittest.mock import MagicMock

from tinker_cookbook.recipes.multiplayer_rl.debate.dataset import DebateDataset
from tinker_cookbook.recipes.multiplayer_rl.debate.plugins import format_penalty_reward_fn
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateGameSpec,
    DebateProblemSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    Utterance,
)


# -- TODO 1: format_penalty_reward_fn --


def test_format_penalty_fields_none():
    """Format violation (fields=None) returns -0.1."""
    utt = Utterance(
        role=Role.DEBATER_A,
        round_index=0,
        phase=Phase.PROPOSE,
        text="some text",
        token_count=5,
        slot_id=0,
        fields=None,
    )
    dummy_state = MagicMock(spec=DebateState)
    result = format_penalty_reward_fn(dummy_state, dummy_state, Role.DEBATER_A, utt)
    assert result == -0.1


def test_format_penalty_fields_present():
    """Valid fields returns 0.0."""
    utt = Utterance(
        role=Role.DEBATER_A,
        round_index=0,
        phase=Phase.PROPOSE,
        text="some text",
        token_count=5,
        slot_id=0,
        fields={"answer": "A"},
    )
    dummy_state = MagicMock(spec=DebateState)
    result = format_penalty_reward_fn(dummy_state, dummy_state, Role.DEBATER_A, utt)
    assert result == 0.0


def test_format_penalty_no_utterance():
    """No utterance (None) returns 0.0."""
    dummy_state = MagicMock(spec=DebateState)
    result = format_penalty_reward_fn(dummy_state, dummy_state, Role.DEBATER_A, None)
    assert result == 0.0


# -- TODO 2: inter-epoch shuffle --

_SEQ_1 = DebateGameSpec(protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1)


def _make_problems(n: int) -> list[DebateProblemSpec]:
    return [
        DebateProblemSpec.from_seat_answers(f"q{i}", f"a{i}", f"b{i}", ScoringMode.MCQ)
        for i in range(n)
    ]


def _get_batch_prompts(ds: DebateDataset, index: int) -> list[str]:
    batch = ds.get_batch(index)
    return [b.problem.task_prompt for b in batch]  # type: ignore[union-attr]


def test_shuffle_different_epochs():
    """Different epochs should (very likely) produce different orderings."""
    problems = _make_problems(8)
    ds = DebateDataset(
        problems=problems,
        batch_size=4,
        group_size=1,
        game=_SEQ_1,
        renderer=MagicMock(),
        shuffle_seed=42,
    )
    # len(ds) = ceil(8/4) = 2 batches per epoch
    assert len(ds) == 2

    # Epoch 0: batches 0, 1
    epoch0_batch0 = _get_batch_prompts(ds, 0)
    epoch0_batch1 = _get_batch_prompts(ds, 1)

    # Epoch 1: batches 2, 3
    epoch1_batch0 = _get_batch_prompts(ds, 2)
    epoch1_batch1 = _get_batch_prompts(ds, 3)

    # All problems should still be present in each epoch
    epoch0_all = epoch0_batch0 + epoch0_batch1
    epoch1_all = epoch1_batch0 + epoch1_batch1
    assert set(epoch0_all) == {f"q{i}" for i in range(8)}
    assert set(epoch1_all) == {f"q{i}" for i in range(8)}

    # The orderings should differ between epochs (with overwhelming probability)
    assert epoch0_all != epoch1_all


def test_shuffle_deterministic_same_seed():
    """Same seed produces same shuffle sequence."""
    problems = _make_problems(6)

    ds1 = DebateDataset(
        problems=problems,
        batch_size=3,
        group_size=1,
        game=_SEQ_1,
        renderer=MagicMock(),
        shuffle_seed=123,
    )
    ds2 = DebateDataset(
        problems=problems,
        batch_size=3,
        group_size=1,
        game=_SEQ_1,
        renderer=MagicMock(),
        shuffle_seed=123,
    )

    for i in range(4):
        assert _get_batch_prompts(ds1, i) == _get_batch_prompts(ds2, i)
