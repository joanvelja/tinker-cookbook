"""Dataset builders for RLVR recipes.

Top half: base builder + math/gsm8k builders (Worker C).
Worker D continues below the marker comment.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import cast

import chz
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.recipes.rlvr.env import (
    ANSWER_FORMAT_INSTRUCTION,
    BOXED_FORMAT_INSTRUCTION,
    RLVRDataset,
    extract_final_answer,
)
from tinker_cookbook.recipes.rlvr.graders import GraderConfig, LLMGraderConfig, SympyGraderConfig
from tinker_cookbook.recipes.rlvr.types import ExtractFn, RLVRExample
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fewshot helpers
# ---------------------------------------------------------------------------

_BOXED_FORMAT_INSTRUCTION = " Write your answer in \\boxed{} format."


def _standard_fewshot_prefix() -> list[renderers.Message]:
    """Strawberry-counting fewshot pair (same as MathEnv.standard_fewshot_prefix)."""
    return [
        {
            "role": "user",
            "content": "How many r's are in strawberry?" + _BOXED_FORMAT_INSTRUCTION,
        },
        {
            "role": "assistant",
            "content": (
                "Let's spell the word out and number all the letters: "
                "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
                "We have r's at positions 3, 8, and 9. \\boxed{3}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Base builder
# ---------------------------------------------------------------------------


@chz.chz
class RLVRDatasetBuilder(RLDatasetBuilder):
    """Abstract base for all RLVR dataset builders.

    Subclasses override ``_load_data`` and optionally ``_get_extract_fn`` /
    ``_resolve_convo_prefix``.
    """

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    grader_config: GraderConfig  # abstract — no default
    seed: int = 42
    format_instruction: str = ANSWER_FORMAT_INSTRUCTION
    convo_prefix: list[renderers.Message] | None = None
    n_batches: int | None = None
    dataset_name: str = "rlvr"
    format_coef: float = 0.1
    eos_coef: float = 0.0

    @abstractmethod
    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        """Return (train_examples, optional_eval_examples)."""
        ...

    def _get_extract_fn(self) -> ExtractFn:
        return extract_final_answer

    def _resolve_convo_prefix(self) -> list[renderers.Message] | None:
        return self.convo_prefix

    async def __call__(self) -> tuple[RLVRDataset, RLVRDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        grader = self.grader_config.build(
            concurrency_hint=self.batch_size * self.group_size,
        )
        extract_fn = self._get_extract_fn()
        convo_prefix = self._resolve_convo_prefix()

        train_examples, eval_examples = self._load_data()

        train_ds = RLVRDataset(
            examples=train_examples,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            grader=grader,
            extract_fn=extract_fn,
            format_instruction=self.format_instruction,
            convo_prefix=convo_prefix,
            dataset_name=self.dataset_name,
            n_batches=self.n_batches,
            format_coef=self.format_coef,
            eos_coef=self.eos_coef,
        )

        eval_ds: RLVRDataset | None = None
        if eval_examples is not None:
            eval_ds = RLVRDataset(
                examples=eval_examples,
                batch_size=self.batch_size,
                group_size=1,
                renderer=renderer,
                grader=grader,
                extract_fn=extract_fn,
                format_instruction=self.format_instruction,
                convo_prefix=convo_prefix,
                dataset_name=f"{self.dataset_name}_eval",
                n_batches=None,
                format_coef=self.format_coef,
                eos_coef=self.eos_coef,
            )

        return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Sympy + boxed base
# ---------------------------------------------------------------------------


@chz.chz
class SympyBoxedBuilder(RLVRDatasetBuilder):
    """Builder base for datasets graded with sympy and using \\boxed{} extraction."""

    grader_config: GraderConfig = SympyGraderConfig()
    format_instruction: str = _BOXED_FORMAT_INSTRUCTION
    include_fewshot: bool = True
    seed: int = 0  # math builders historically used seed=0

    def _get_extract_fn(self) -> ExtractFn:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        return extract_boxed

    def _resolve_convo_prefix(self) -> list[renderers.Message] | None:
        if self.include_fewshot:
            return _standard_fewshot_prefix()
        return self.convo_prefix

    @abstractmethod
    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]: ...


# ---------------------------------------------------------------------------
# MATH (Hendrycks)
# ---------------------------------------------------------------------------


@chz.chz
class MathBuilder(SympyBoxedBuilder):
    dataset_name: str = "math"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        # Load test split (MATH-500)
        test_dataset = cast(
            Dataset,
            load_dataset("HuggingFaceH4/MATH-500", name="default", split="test"),
        )
        test_problems: set[str] = {row["problem"] for row in test_dataset}  # type: ignore[arg-type]

        test_examples: list[RLVRExample] = []
        for row in test_dataset:
            try:
                ref = extract_boxed(row["solution"])  # type: ignore[index]
            except ValueError:
                continue
            test_examples.append(RLVRExample(question=row["problem"], reference=ref))  # type: ignore[index]

        # Load train split (all Hendrycks configs, minus contamination)
        dataset_name = "EleutherAI/hendrycks_math"
        configs = get_dataset_config_names(dataset_name)
        pieces: list[Dataset] = []
        for cfg in configs:
            for split in ("train", "test"):
                ds = load_dataset(dataset_name, name=cfg, split=split)
                ds = ds.filter(lambda ex: ex["problem"] not in test_problems)
                pieces.append(ds)
        full_train = concatenate_datasets(pieces).shuffle(seed=self.seed)

        train_examples: list[RLVRExample] = []
        for row in full_train:
            try:
                ref = extract_boxed(row["solution"])  # type: ignore[index]
            except ValueError:
                logger.warning("Skipping MATH row with unparseable solution")
                continue
            train_examples.append(RLVRExample(question=row["problem"], reference=ref))  # type: ignore[index]

        return train_examples, test_examples


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K solution field.

    GSM8K format places the answer on a line starting with ``####``.
    """
    import re

    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


_GSM8K_FORMAT_INSTRUCTION = " Provide a numerical answer without units, written inside \\boxed{}."


@chz.chz
class Gsm8kBuilder(SympyBoxedBuilder):
    dataset_name: str = "gsm8k"
    format_instruction: str = _GSM8K_FORMAT_INSTRUCTION

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        train_ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="train"))
        test_ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="test"))

        train_ds = train_ds.shuffle(seed=self.seed)

        train_examples: list[RLVRExample] = []
        for row in train_ds:
            try:
                ref = _extract_gsm8k_final_answer(row["answer"])  # type: ignore[index]
            except ValueError:
                logger.warning("Skipping GSM8K train row with unparseable answer")
                continue
            train_examples.append(RLVRExample(question=row["question"], reference=ref))  # type: ignore[index]

        test_examples: list[RLVRExample] = []
        for row in test_ds:
            try:
                ref = _extract_gsm8k_final_answer(row["answer"])  # type: ignore[index]
            except ValueError:
                logger.warning("Skipping GSM8K test row with unparseable answer")
                continue
            test_examples.append(RLVRExample(question=row["question"], reference=ref))  # type: ignore[index]

        return train_examples, test_examples


# ---------------------------------------------------------------------------
# Polaris
# ---------------------------------------------------------------------------


@chz.chz
class PolarisBuilder(SympyBoxedBuilder):
    include_fewshot: bool = False
    dataset_name: str = "polaris"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        import random

        ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
        rows = [ds[i] for i in range(len(ds))]
        random.Random(self.seed).shuffle(rows)

        train_examples: list[RLVRExample] = []
        for x in rows:
            problem = x["problem"]  # type: ignore[index]
            answer = x["answer"]  # type: ignore[index]
            if problem and answer:
                train_examples.append(RLVRExample(question=str(problem), reference=str(answer)))

        return train_examples, None


# ---------------------------------------------------------------------------
# DeepMath
# ---------------------------------------------------------------------------


@chz.chz
class DeepMathBuilder(SympyBoxedBuilder):
    include_fewshot: bool = False
    dataset_name: str = "deepmath"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        import random

        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        rows = [ds[i] for i in range(len(ds))]
        random.Random(self.seed).shuffle(rows)

        train_examples: list[RLVRExample] = []
        for x in rows:
            question = x["question"]  # type: ignore[index]
            answer = x["final_answer"]  # type: ignore[index]
            if question and answer:
                train_examples.append(RLVRExample(question=str(question), reference=str(answer)))

        return train_examples, None


# ---------------------------------------------------------------------------
# GPQA Open-Ended (LLM-graded)
# ---------------------------------------------------------------------------


@chz.chz
class GpqaOpenEndedBuilder(RLVRDatasetBuilder):
    grader_config: GraderConfig = LLMGraderConfig()
    subset: str = "extended"
    split: str = "train"
    eval_frac: float = 0.1
    dataset_name: str = "gpqa_oe"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        from tinker_cookbook.recipes.multiplayer_rl.debate.data.gpqa import (
            load_gpqa_open_ended_rows,
        )

        raw = load_gpqa_open_ended_rows(
            subset=self.subset, split=self.split, seed=self.seed,
        )
        rows = [
            RLVRExample(question=str(r["question"]), reference=str(r["answer"]))
            for r in raw
        ]

        n_eval = max(1, int(len(rows) * self.eval_frac))
        train = rows[n_eval:]
        eval_ = rows[:n_eval]
        return train, eval_


# ---------------------------------------------------------------------------
# OmniMath (LLM-graded, boxed extraction)
# ---------------------------------------------------------------------------

_MATH_GRADER_SYSTEM = """\
Grade: CORRECT or INCORRECT.
CORRECT = mathematically equivalent (notation/format differences are fine).
INCORRECT = mathematically different (even subtly: wrong number, sign, missing factor).
Examples:
  CORRECT: '\\frac{1}{2}' vs '0.5' — same value.
  CORRECT: '3\\pi' vs '3π' — same expression.
  INCORRECT: '\\frac{1}{2}' vs '\\frac{1}{3}' — different fraction.
  INCORRECT: 'x^2 + 1' vs 'x^2 - 1' — different sign.
One word."""


@chz.chz
class OmniMathBuilder(RLVRDatasetBuilder):
    grader_config: GraderConfig = LLMGraderConfig(system_prompt=_MATH_GRADER_SYSTEM)
    format_instruction: str = BOXED_FORMAT_INSTRUCTION
    eval_frac: float = 0.15
    dataset_name: str = "omni_math"

    def _get_extract_fn(self) -> ExtractFn:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        return extract_boxed

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        import random

        ds = load_dataset("martheballon/Omni-MATH-2", split="train")
        rows_raw = [ds[i] for i in range(len(ds))]
        random.Random(self.seed).shuffle(rows_raw)

        rows = [
            RLVRExample(question=str(r["problem"]), reference=str(r["answer"]))
            for r in rows_raw
        ]

        n_eval = max(1, int(len(rows) * self.eval_frac))
        train = rows[n_eval:]
        eval_ = rows[:n_eval]
        return train, eval_


# ---------------------------------------------------------------------------
# Builder registry
# ---------------------------------------------------------------------------

DATASET_BUILDER_MAP: dict[str, type[RLVRDatasetBuilder]] = {
    "math": MathBuilder,
    "gsm8k": Gsm8kBuilder,
    "polaris": PolarisBuilder,
    "deepmath": DeepMathBuilder,
    "gpqa_oe": GpqaOpenEndedBuilder,
    "omni_math": OmniMathBuilder,
}
