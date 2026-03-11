"""Dataset adapters for debate evaluation via Inspect AI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from inspect_ai.dataset import Sample

from ..data.gpqa import (
    assign_seat_answers,
    load_gpqa_mcq_problems,
    load_gpqa_open_ended_problems,
    problem_to_sample,
)
from ..scoring.mcq import normalize_mcq
from ..types import DebateProblemSpec, ScoringMode


@runtime_checkable
class DatasetAdapter(Protocol):
    """Converts an external dataset to Inspect Samples for debate eval."""

    def to_samples(self) -> list[Sample]: ...

    def resolve_scoring_mode(self) -> ScoringMode: ...


def infer_scoring_mode_from_samples(samples: Sequence[Sample]) -> ScoringMode:
    targets: list[str] = []
    for sample in samples:
        if sample.target is None:
            continue
        target_text = sample.target.text if hasattr(sample.target, "text") else str(sample.target)
        targets.append(target_text)
    if targets and all(normalize_mcq(target) is not None for target in targets):
        return ScoringMode.MCQ
    return ScoringMode.OPEN_ENDED


def resolve_adapter_scoring_mode(
    adapter: DatasetAdapter,
    *,
    samples: Sequence[Sample] | None = None,
) -> ScoringMode:
    resolved_samples = samples if samples is not None else adapter.to_samples()
    resolver = getattr(adapter, "resolve_scoring_mode", None)
    if callable(resolver):
        mode = resolver()
        if isinstance(mode, ScoringMode):
            return mode
    return infer_scoring_mode_from_samples(resolved_samples)


# ---------------------------------------------------------------------------
# ProblemsAdapter — wraps pre-loaded problems (no HF re-fetch)
# ---------------------------------------------------------------------------


class ProblemsAdapter:
    """Adapts pre-loaded DebateProblemSpecs to the DatasetAdapter protocol.

    Use this to build eval adapters from a known train/test split,
    eliminating contamination from independent HF re-loads.
    """

    def __init__(
        self,
        problems: Sequence[DebateProblemSpec],
        *,
        source: str = "problems",
        limit: int | None = None,
    ) -> None:
        self._problems = list(problems[:limit] if limit is not None else problems)
        self._source = source

    def to_samples(self) -> list[Sample]:
        return [problem_to_sample(p, source=self._source) for p in self._problems]

    def resolve_scoring_mode(self) -> ScoringMode:
        if not self._problems:
            return ScoringMode.MCQ
        return self._problems[0].scoring_mode


# ---------------------------------------------------------------------------
# HF-loading adapters — for standalone eval (no training, no contamination)
# ---------------------------------------------------------------------------


class GPQAAdapter:
    """Loads GPQA MCQ from HuggingFace and creates Samples.

    For standalone eval only. Training-loop eval should use ProblemsAdapter
    with test_problems from ProblemSource.load().
    """

    def __init__(
        self,
        subset: str = "gpqa_diamond",
        limit: int | None = None,
        seed: int = 42,
        free_debate: bool = False,
    ) -> None:
        self._subset = subset
        self._limit = limit
        self._seed = seed
        self._free_debate = free_debate

    def to_samples(self) -> list[Sample]:
        problems = load_gpqa_mcq_problems(
            n=self._limit, subset=self._subset, seed=self._seed
        )
        if not self._free_debate:
            problems = assign_seat_answers(problems, seed=self._seed)
        return [problem_to_sample(p, source="gpqa_diamond") for p in problems]

    def resolve_scoring_mode(self) -> ScoringMode:
        return ScoringMode.MCQ


class GPQAOpenEndedAdapter:
    """Loads GPQA open-ended from HuggingFace and creates Samples.

    For standalone eval only. Training-loop eval should use ProblemsAdapter
    with test_problems from ProblemSource.load().
    """

    def __init__(
        self,
        subset: str = "extended",
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
        record_ids: list[str] | None = None,
    ) -> None:
        self._subset = subset
        self._split = split
        self._limit = limit
        self._seed = seed
        self._record_ids = list(record_ids) if record_ids is not None else None

    def to_samples(self) -> list[Sample]:
        problems = load_gpqa_open_ended_problems(
            subset=self._subset,
            split=self._split,
            seed=self._seed,
            record_ids=self._record_ids,
            limit=self._limit,
        )
        return [problem_to_sample(p, source="gpqa_open_ended") for p in problems]

    def resolve_scoring_mode(self) -> ScoringMode:
        return ScoringMode.OPEN_ENDED
