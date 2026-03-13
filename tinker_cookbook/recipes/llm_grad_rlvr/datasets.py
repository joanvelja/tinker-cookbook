"""Concrete dataset builders for LLM-graded RLVR.

Each builder is a thin subclass of LLMGradedDatasetBuilder that overrides:
  _load_rows()       — how to fetch and project the data
  _get_extract_fn()  — how to parse the model's answer (if not <final_answer> tags)
"""

import chz

from tinker_cookbook.recipes.llm_grad_rlvr.env import (
    BOXED_FORMAT_INSTRUCTION,
    ExtractFn,
    LLMGradedDatasetBuilder,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


@chz.chz
class GpqaOpenEndedBuilder(LLMGradedDatasetBuilder):
    subset: str = "extended"
    split: str = "train"
    dataset_name: str = "gpqa_oe"

    def _load_rows(self) -> list[dict[str, str]]:
        from tinker_cookbook.recipes.multiplayer_rl.debate.data.gpqa import (
            load_gpqa_open_ended_rows,
        )

        raw = load_gpqa_open_ended_rows(subset=self.subset, split=self.split, seed=self.seed)
        return [{"question": str(r["question"]), "answer": str(r["answer"])} for r in raw]


@chz.chz
class OmniMathBuilder(LLMGradedDatasetBuilder):
    dataset_name: str = "omni_math"
    answer_format_instruction: str = BOXED_FORMAT_INSTRUCTION

    def _get_extract_fn(self) -> ExtractFn:
        return extract_boxed

    def _load_rows(self) -> list[dict[str, str]]:
        import random

        from datasets import load_dataset

        ds = load_dataset("martheballon/Omni-MATH-2", split="train")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
        random.Random(self.seed).shuffle(rows)
        return rows
