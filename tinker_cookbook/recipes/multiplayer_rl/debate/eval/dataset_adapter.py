"""Dataset adapters for debate evaluation via Inspect AI."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from inspect_ai.dataset import Sample

from ..scoring.mcq import normalize_mcq
from ..types import ScoringMode


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


class GPQAAdapter:
    """Loads GPQA diamond and creates Samples for MCQ debate evaluation.

    Each Sample:
      input = task_prompt (question + shuffled ABCD choices)
      target = correct answer label (A/B/C/D)
      metadata = {"answer_a": correct_label, "answer_b": wrong_label, "source": "gpqa_diamond"}
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
        import datasets as hf_datasets

        ds = hf_datasets.load_dataset("Idavidrein/gpqa", self._subset, split="train")
        rng = random.Random(self._seed)

        n = len(ds) if self._limit is None else min(self._limit, len(ds))
        indices = rng.sample(range(len(ds)), n)

        samples: list[Sample] = []
        for idx in indices:
            row = ds[idx]
            correct = row["Correct Answer"]
            wrong = [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]

            # Shuffle into ABCD, track correct label.
            options = [correct] + wrong
            rng.shuffle(options)
            target_label = chr(ord("A") + options.index(correct))

            # Format as MCQ prompt.
            question = row["Question"]
            option_lines = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
            task_prompt = f"{question}\n\n{option_lines}"

            if self._free_debate:
                metadata = {"answer_a": "", "answer_b": "", "source": "gpqa_diamond"}
            else:
                # Debater A gets correct, B gets random wrong.
                wrong_label = rng.choice(
                    [chr(ord("A") + i) for i in range(4) if chr(ord("A") + i) != target_label]
                )
                metadata = {
                    "answer_a": target_label,
                    "answer_b": wrong_label,
                    "source": "gpqa_diamond",
                }

            samples.append(
                Sample(
                    input=task_prompt,
                    target=target_label,
                    metadata=metadata,
                )
            )

        return samples

    def resolve_scoring_mode(self) -> ScoringMode:
        return ScoringMode.MCQ


class GPQAOpenEndedAdapter:
    """Loads GPQA open-ended and creates Samples for semantic debate evaluation.

    Each Sample:
      input = question text
      target = gold free-form answer
      metadata = {
          "answer_a": "",
          "answer_b": "",
          "source": "gpqa_open_ended",
          "record_id": ...,
          "domain": ...,
          "subdomain": ...,
      }
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
        import datasets as hf_datasets

        ds = hf_datasets.load_dataset("joanvelja/gpqa-open-ended", self._subset, split=self._split)
        rows = [ds[i] for i in range(len(ds))]

        if self._record_ids:
            requested = set(self._record_ids)
            rows = [row for row in rows if row.get("record_id") in requested]
            found = {row.get("record_id") for row in rows}
            missing = requested - found
            if missing:
                missing_csv = ", ".join(sorted(str(record_id) for record_id in missing))
                raise ValueError(
                    "GPQA open-ended adapter could not find requested record_ids: "
                    f"{missing_csv}"
                )
            rows.sort(key=lambda row: self._record_ids.index(str(row["record_id"])))
        elif self._limit is not None:
            rng = random.Random(self._seed)
            n = min(self._limit, len(rows))
            rows = [rows[idx] for idx in rng.sample(range(len(rows)), n)]

        samples: list[Sample] = []
        for row in rows:
            samples.append(
                Sample(
                    input=str(row["question"]),
                    target=str(row["answer"]),
                    metadata={
                        "answer_a": "",
                        "answer_b": "",
                        "source": "gpqa_open_ended",
                        "record_id": str(row.get("record_id", "")),
                        "domain": str(row.get("domain", "")),
                        "subdomain": str(row.get("subdomain", "")),
                        "writer_difficulty": row.get("writer_difficulty"),
                        "expert_accuracy": row.get("expert_accuracy"),
                        "non_expert_accuracy": row.get("non_expert_accuracy"),
                        "conversion_type": row.get("conversion_type"),
                        "flag": row.get("flag"),
                    },
                )
            )

        return samples

    def resolve_scoring_mode(self) -> ScoringMode:
        return ScoringMode.OPEN_ENDED
