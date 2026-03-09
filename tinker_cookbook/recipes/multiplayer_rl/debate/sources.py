"""Dataset sources for debate training.

A ``ProblemSource`` returns ``(train, test)`` lists of ``DebateProblemSpec``.
Subclass and register with ``@chz.chz`` to add new datasets — chz subclass
dispatch lets the CLI select the source by name.

Example CLI usage::

    uv run python -m ... \\
        --problem_source=GPQAProblemSource \\
        --problem_source.subset=gpqa_diamond

    uv run python -m ... \\
        --problem_source=GPQAOpenEndedProblemSource \\
        --scorer_builder=DebateScorerBuilder \\
        --scorer_builder.model=openai/gpt-oss-120b
"""

from __future__ import annotations

from abc import abstractmethod

import chz

from .data.gpqa import (
    load_gpqa_mcq_rows,
    load_gpqa_open_ended_rows,
    mcq_row_to_problem,
    open_ended_row_to_problem,
)
from .types import DebateProblemSpec, ScoringMode

import random


@chz.chz
class ProblemSource:
    """Source of debate problems for training. Subclass for new datasets."""

    @abstractmethod
    def load(self) -> tuple[list[DebateProblemSpec], list[DebateProblemSpec]]:
        """Return (train_problems, test_problems)."""
        ...

    @abstractmethod
    def scoring_mode(self) -> ScoringMode:
        """The scoring mode all problems from this source use."""
        ...

    @abstractmethod
    def default_prompts_ref(self) -> str:
        """Default prompts_ref for this source type."""
        ...


@chz.chz
class GPQAProblemSource(ProblemSource):
    """GPQA MCQ problems for free debate (blank seat answers, target tracks gold)."""

    subset: str = "gpqa_extended"
    test_fraction: float = 0.1
    seed: int = 42

    def load(self) -> tuple[list[DebateProblemSpec], list[DebateProblemSpec]]:
        rows = load_gpqa_mcq_rows(subset=self.subset, seed=self.seed)
        n_test = max(1, int(len(rows) * self.test_fraction))

        rng = random.Random(self.seed)
        train = [mcq_row_to_problem(row, rng) for row in rows[:-n_test]]
        test = [mcq_row_to_problem(row, rng) for row in rows[-n_test:]]
        return train, test

    def scoring_mode(self) -> ScoringMode:
        return ScoringMode.MCQ

    def default_prompts_ref(self) -> str:
        return "judge_exploit"


@chz.chz
class GPQAOpenEndedProblemSource(ProblemSource):
    """GPQA open-ended problems (free-form target, no seat answers)."""

    subset: str = "extended"
    split: str = "train"
    test_fraction: float = 0.1
    seed: int = 42
    record_ids: list[str] | None = None

    def load(self) -> tuple[list[DebateProblemSpec], list[DebateProblemSpec]]:
        rows = load_gpqa_open_ended_rows(
            subset=self.subset,
            split=self.split,
            seed=self.seed,
            record_ids=self.record_ids,
        )
        n_test = max(1, int(len(rows) * self.test_fraction))
        train_rows, test_rows = rows[:-n_test], rows[-n_test:]
        return (
            [open_ended_row_to_problem(r) for r in train_rows],
            [open_ended_row_to_problem(r) for r in test_rows],
        )

    def scoring_mode(self) -> ScoringMode:
        return ScoringMode.OPEN_ENDED

    def default_prompts_ref(self) -> str:
        return "open_balanced"
