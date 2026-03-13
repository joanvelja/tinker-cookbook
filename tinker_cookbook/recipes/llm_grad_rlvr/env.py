"""LLM-graded RLVR environment and dataset infrastructure."""

import logging
import math
import re
from functools import partial
from typing import Callable, Sequence

import chz

from tinker_cookbook import renderers
from tinker_cookbook.recipes.llm_grad_rlvr.grader import AsyncLLMGrader, GradingError, LLMGraderConfig
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction functions: pull the final answer out of model output.
# Each raises ValueError if the format wasn't followed.
# ---------------------------------------------------------------------------

ExtractFn = Callable[[str], str]

_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)

ANSWER_FORMAT_INSTRUCTION = "\n\nWrite your final answer inside <final_answer></final_answer> tags."
BOXED_FORMAT_INSTRUCTION = "\n\nWrite your final answer as \\boxed{answer}."


def extract_final_answer(content: str) -> str:
    """Extract content from <final_answer>...</final_answer> tags. Raises ValueError on failure."""
    m = _FINAL_ANSWER_RE.search(content)
    if not m:
        raise ValueError("No <final_answer> tags found")
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LLMGradedEnv(ProblemEnv):
    def __init__(
        self,
        question: str,
        answer: str,
        renderer: renderers.Renderer,
        grader: AsyncLLMGrader,
        extract_fn: ExtractFn = extract_final_answer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        answer_format_instruction: str = "",
    ):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.question = question
        self.answer = answer
        self.grader = grader
        self.extract_fn = extract_fn
        self.answer_format_instruction = answer_format_instruction

    def get_question(self) -> str:
        return self.question + self.answer_format_instruction

    async def check_answer(self, sample_str: str) -> bool:
        try:
            extracted = self.extract_fn(sample_str)
        except ValueError:
            return False
        try:
            return await self.grader.grade(self.question, self.answer, extracted)
        except GradingError:
            return False

    def check_format(self, sample_str: str) -> bool:
        try:
            self.extract_fn(sample_str)
            return True
        except ValueError:
            return False

    def get_reference_answer(self) -> str:
        return self.answer


# ---------------------------------------------------------------------------
# Dataset + builder base
# ---------------------------------------------------------------------------


class LLMGradedDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        grader: AsyncLLMGrader,
        extract_fn: ExtractFn = extract_final_answer,
        convo_prefix: list[renderers.Message] | None = None,
        answer_format_instruction: str = "",
        dataset_name: str = "llm_graded",
        n_batches: int | None = None,
    ):
        self._rows = rows
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.grader = grader
        self.extract_fn = extract_fn
        self.convo_prefix = convo_prefix
        self.answer_format_instruction = answer_format_instruction
        self.dataset_name = dataset_name
        self._n_batches = n_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        n = len(self._rows)
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    LLMGradedEnv,
                    question=self._rows[(index * self.batch_size + i) % n]["question"],
                    answer=self._rows[(index * self.batch_size + i) % n]["answer"],
                    renderer=self.renderer,
                    grader=self.grader,
                    extract_fn=self.extract_fn,
                    convo_prefix=self.convo_prefix,
                    answer_format_instruction=self.answer_format_instruction,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        if self._n_batches is not None:
            return self._n_batches
        return math.ceil(len(self._rows) / self.batch_size)


@chz.chz
class LLMGradedDatasetBuilder(RLDatasetBuilder):
    """Base builder for LLM-graded RLVR datasets.

    Subclasses override:
      _load_rows()       — returns [{"question": ..., "answer": ...}, ...]
      _get_extract_fn()  — returns the extraction function (default: <final_answer> tags)
    """

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    seed: int = 42
    grader_config: LLMGraderConfig = LLMGraderConfig()
    eval_frac: float = 0.1
    convo_prefix: list[renderers.Message] | None = None
    answer_format_instruction: str = ANSWER_FORMAT_INSTRUCTION
    dataset_name: str = "llm_graded"
    n_batches: int | None = None

    def _load_rows(self) -> list[dict[str, str]]:
        raise NotImplementedError

    def _get_extract_fn(self) -> ExtractFn:
        return extract_final_answer

    def _make_dataset(
        self,
        rows: list[dict[str, str]],
        renderer: renderers.Renderer,
        grader: AsyncLLMGrader,
        extract_fn: ExtractFn,
        group_size: int,
        dataset_name: str,
        n_batches: int | None = None,
    ) -> LLMGradedDataset:
        return LLMGradedDataset(
            rows=rows,
            batch_size=self.batch_size,
            group_size=group_size,
            renderer=renderer,
            grader=grader,
            extract_fn=extract_fn,
            convo_prefix=self.convo_prefix,
            answer_format_instruction=self.answer_format_instruction,
            dataset_name=dataset_name,
            n_batches=n_batches,
        )

    async def __call__(self) -> tuple[LLMGradedDataset, LLMGradedDataset | None]:
        rows = self._load_rows()
        n_eval = max(1, int(len(rows) * self.eval_frac))
        train_rows = rows[n_eval:]
        eval_rows = rows[:n_eval]

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        grader_config = self.grader_config
        if grader_config.client.max_concurrent is None:
            new_client = chz.replace(
                grader_config.client, max_concurrent=self.batch_size * self.group_size
            )
            grader_config = chz.replace(grader_config, client=new_client)
        grader = AsyncLLMGrader(grader_config)

        extract_fn = self._get_extract_fn()
        train_ds = self._make_dataset(
            train_rows, renderer, grader, extract_fn,
            group_size=self.group_size,
            dataset_name=self.dataset_name,
            n_batches=self.n_batches,
        )
        eval_ds = self._make_dataset(
            eval_rows, renderer, grader, extract_fn,
            group_size=1,
            dataset_name=f"{self.dataset_name}_eval",
        )
        return train_ds, eval_ds
