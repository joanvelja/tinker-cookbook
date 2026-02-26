"""Dataset adapters for debate evaluation via Inspect AI."""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from inspect_ai.dataset import Sample


@runtime_checkable
class DatasetAdapter(Protocol):
    """Converts an external dataset to Inspect Samples for debate eval."""
    def to_samples(self) -> list[Sample]: ...


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
    ) -> None:
        self._subset = subset
        self._limit = limit
        self._seed = seed

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
            option_lines = "\n".join(
                f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
            )
            task_prompt = f"{question}\n\n{option_lines}"

            # Debater A gets correct, B gets random wrong.
            wrong_label = rng.choice(
                [chr(ord("A") + i) for i in range(4) if chr(ord("A") + i) != target_label]
            )

            samples.append(
                Sample(
                    input=task_prompt,
                    target=target_label,
                    metadata={
                        "answer_a": target_label,
                        "answer_b": wrong_label,
                        "source": "gpqa_diamond",
                    },
                )
            )

        return samples
