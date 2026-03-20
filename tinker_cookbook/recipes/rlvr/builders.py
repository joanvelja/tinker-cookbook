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
    GPQA_FORMAT_INSTRUCTION,
    RLVRDataset,
    extract_final_answer,
)
from tinker_cookbook.recipes.rlvr.graders import CompositeGraderConfig, GraderConfig, LLMGraderConfig, SympyGraderConfig
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
# Shared data-loading helpers
# ---------------------------------------------------------------------------


def _extract_examples(
    dataset: Dataset,
    question_field: str,
    answer_field: str,
    extract_fn: ExtractFn | None = None,
    *,
    warn_prefix: str = "",
) -> list[RLVRExample]:
    """Convert a HF dataset to RLVRExamples, optionally applying extract_fn to parse answers.

    If *extract_fn* is given, rows where it raises ValueError are skipped (with a warning
    if *warn_prefix* is set). If *extract_fn* is None, the raw answer field is used directly.
    """
    examples: list[RLVRExample] = []
    for row in dataset:
        question = row[question_field]  # type: ignore[index]
        raw_answer = row[answer_field]  # type: ignore[index]
        if not question or not raw_answer:
            continue
        if extract_fn is not None:
            try:
                ref = extract_fn(str(raw_answer))
            except ValueError:
                if warn_prefix:
                    logger.warning("Skipping %s row with unparseable answer", warn_prefix)
                continue
        else:
            ref = str(raw_answer)
        examples.append(RLVRExample(question=str(question), reference=ref))
    return examples


def _load_shuffle_split(
    dataset_path: str,
    question_field: str,
    answer_field: str,
    seed: int,
    eval_frac: float = 0.0,
) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
    """Load a single-split HF dataset, shuffle, convert, and optionally carve out an eval set."""
    import random

    ds = load_dataset(dataset_path, split="train")
    rows_raw = [ds[i] for i in range(len(ds))]
    random.Random(seed).shuffle(rows_raw)

    rows = [
        RLVRExample(question=str(r[question_field]), reference=str(r[answer_field]))
        for r in rows_raw
        if r[question_field] and r[answer_field]  # type: ignore[index]
    ]

    if eval_frac > 0:
        n_eval = max(1, int(len(rows) * eval_frac))
        return rows[n_eval:], rows[:n_eval]
    return rows, None


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
    grade_full_response: bool = False
    gt_scorer_config: GraderConfig | None = None

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
        gt_scorer = (
            self.gt_scorer_config.build(concurrency_hint=self.batch_size * self.group_size)
            if self.gt_scorer_config is not None
            else None
        )

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
            grade_full_response=self.grade_full_response,
            gt_scorer=gt_scorer,
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
                gt_scorer=gt_scorer,
                grade_full_response=self.grade_full_response,
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
        test_examples = _extract_examples(test_dataset, "problem", "solution", extract_boxed)

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
        train_examples = _extract_examples(
            full_train, "problem", "solution", extract_boxed, warn_prefix="MATH",
        )

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

        extract = _extract_gsm8k_final_answer
        train_examples = _extract_examples(train_ds, "question", "answer", extract, warn_prefix="GSM8K")
        test_examples = _extract_examples(test_ds, "question", "answer", extract, warn_prefix="GSM8K")
        return train_examples, test_examples


# ---------------------------------------------------------------------------
# Polaris
# ---------------------------------------------------------------------------


@chz.chz
class PolarisBuilder(SympyBoxedBuilder):
    include_fewshot: bool = False
    dataset_name: str = "polaris"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        return _load_shuffle_split(
            "POLARIS-Project/Polaris-Dataset-53K", "problem", "answer", self.seed,
        )


# ---------------------------------------------------------------------------
# DeepMath
# ---------------------------------------------------------------------------


@chz.chz
class DeepMathBuilder(SympyBoxedBuilder):
    include_fewshot: bool = False
    dataset_name: str = "deepmath"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        return _load_shuffle_split(
            "zwhe99/DeepMath-103K", "question", "final_answer", self.seed,
        )


# ---------------------------------------------------------------------------
# GPQA Open-Ended (LLM-graded)
# ---------------------------------------------------------------------------


def _gpqa_system() -> list[renderers.Message]:
    return [{"role": "system", "content": "You are solving a graduate-level science question."}]


@chz.chz
class GpqaOpenEndedBuilder(RLVRDatasetBuilder):
    grader_config: GraderConfig = LLMGraderConfig()
    convo_prefix: list[renderers.Message] | None = chz.field(default_factory=_gpqa_system)
    format_instruction: str = GPQA_FORMAT_INSTRUCTION
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
  CORRECT: '\\text{D}' vs 'D' — same answer, LaTeX wrapper is irrelevant.
  CORRECT: 'f(n) = n' vs 'f(x) = x' — same function, variable name doesn't matter.
  CORRECT: '1' vs 'only 1' — same value, prose wrapper is irrelevant.
  CORRECT: 'n composite and n ≠ pq' vs 'All composite integers except products of two distinct primes' — same set.
  INCORRECT: '\\frac{1}{2}' vs '\\frac{1}{3}' — different fraction.
  INCORRECT: 'x^2 + 1' vs 'x^2 - 1' — different sign.
  INCORRECT: '4' vs '5' — different number.
One word."""


@chz.chz
class OmniMathBuilder(RLVRDatasetBuilder):
    grader_config: GraderConfig = CompositeGraderConfig(system_prompt=_MATH_GRADER_SYSTEM)
    format_instruction: str = BOXED_FORMAT_INSTRUCTION
    eval_frac: float = 0.15
    dataset_name: str = "omni_math"

    def _get_extract_fn(self) -> ExtractFn:
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        return extract_boxed

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        return _load_shuffle_split(
            "martheballon/Omni-MATH-2", "problem", "answer", self.seed,
            eval_frac=self.eval_frac,
        )


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
