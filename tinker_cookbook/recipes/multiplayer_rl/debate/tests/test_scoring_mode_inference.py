from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from inspect_ai.dataset import Sample

from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateDataset
from tinker_cookbook.recipes.multiplayer_rl.debate.scripts import smoke_gpqa_open_ended
from tinker_cookbook.recipes.multiplayer_rl.debate.types import ProtocolKind
from tinker_cookbook.recipes.multiplayer_rl.debate.types import ScoringMode


def test_debate_dataset_requires_scorer_for_open_ended_three_tuple_problem() -> None:
    problems = [
        (
            "Which city should host the summit?",
            "Paris",
            "London",
        )
    ]

    with pytest.raises(ValueError, match="OPEN_ENDED debate scoring requires a scorer"):
        DebateDataset(
            problems=problems,
            batch_size=1,
            renderer=MagicMock(),
            protocol_kind=ProtocolKind.SEQUENTIAL,
            num_rounds=1,
        )


def test_gpqa_open_ended_smoke_problem_conversion_preserves_open_ended_mode_for_letter_targets() -> None:
    class _FakeOpenEndedAdapter:
        def to_samples(self) -> list[Sample]:
            return [
                Sample(
                    input="Name the amino acid represented by the single-letter code A.",
                    target="A",
                    metadata={
                        "answer_a": "",
                        "answer_b": "",
                        "source": "gpqa_open_ended",
                        "record_id": "rec-letter",
                    },
                ),
                Sample(
                    input="Why is the sky blue?",
                    target="Rayleigh scattering.",
                    metadata={
                        "answer_a": "",
                        "answer_b": "",
                        "source": "gpqa_open_ended",
                        "record_id": "rec-freeform",
                    },
                ),
            ]

        def resolve_scoring_mode(self) -> ScoringMode:
            return ScoringMode.OPEN_ENDED

    adapter = _FakeOpenEndedAdapter()
    problems = smoke_gpqa_open_ended._samples_to_problems(adapter.to_samples())

    dataset = DebateDataset(
        problems=problems,
        batch_size=len(problems),
        renderer=MagicMock(),
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=1,
        scoring_mode=adapter.resolve_scoring_mode(),
        scorer=MagicMock(),
    )

    assert dataset.scoring_mode == adapter.resolve_scoring_mode()
