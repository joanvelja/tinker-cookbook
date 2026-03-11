"""LLM-graded environment for open-ended QA tasks (e.g. GPQA open-ended)."""

import logging
import math
import re
from functools import partial
from typing import Sequence

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.scoring import (
    AsyncBinaryJudge,
    BinaryJudgeBuilder,
    BinaryJudgeError,
    BinaryJudgeTemplate,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)

ANSWER_FORMAT_INSTRUCTION = "\n\nWrite your final answer inside <final_answer></final_answer> tags."

_DEFAULT_GRADER_TEMPLATE = BinaryJudgeTemplate(
    system="You are a grading assistant. Compare the student's response to the target answer. Respond with exactly one word: CORRECT or INCORRECT.",
    user="Question: {question}\n\nTarget answer: {target}\n\nStudent response: {response}\n\nVerdict:",
    name="correctness",
)


def extract_final_answer(content: str) -> str | None:
    m = _FINAL_ANSWER_RE.search(content)
    return m.group(1).strip() if m else None


class LLMGradedEnv(ProblemEnv):
    def __init__(
        self,
        question: str,
        answer: str,
        renderer: renderers.Renderer,
        judge: AsyncBinaryJudge,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        answer_format_instruction: str = "",
    ):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.question = question
        self.answer = answer
        self.judge = judge
        self.answer_format_instruction = answer_format_instruction

    def get_question(self) -> str:
        return self.question + self.answer_format_instruction

    async def check_answer(self, sample_str: str) -> bool:
        extracted = extract_final_answer(sample_str)
        if extracted is None:
            return False
        try:
            return await self.judge.judge(question=self.question, target=self.answer, response=extracted)
        except BinaryJudgeError:
            return False

    def check_format(self, sample_str: str) -> bool:
        return extract_final_answer(sample_str) is not None

    def get_reference_answer(self) -> str:
        return self.answer


class LLMGradedDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        judge: AsyncBinaryJudge,
        convo_prefix: list[renderers.Message] | None = None,
        answer_format_instruction: str = "",
        dataset_name: str = "llm_graded",
        n_batches: int | None = None,
    ):
        self._rows = rows
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.judge = judge
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
                    judge=self.judge,
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
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    judge_builder: BinaryJudgeBuilder = BinaryJudgeBuilder(provider="openai_compatible", model="gpt-5-mini")
    grader_template: BinaryJudgeTemplate | None = None
    eval_frac: float = 0.1
    convo_prefix: list[renderers.Message] | None = None
    answer_format_instruction: str = ANSWER_FORMAT_INSTRUCTION
    dataset_name: str = "llm_graded"
    n_batches: int | None = None

    def _load_rows(self) -> list[dict[str, str]]:
        raise NotImplementedError

    async def __call__(self) -> tuple[LLMGradedDataset, LLMGradedDataset | None]:
        rows = self._load_rows()
        n_eval = max(1, int(len(rows) * self.eval_frac))
        train_rows = rows[n_eval:]
        eval_rows = rows[:n_eval]

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        client = self.judge_builder.build()
        template = self.grader_template or _DEFAULT_GRADER_TEMPLATE
        judge = AsyncBinaryJudge(client, template)

        train_ds = LLMGradedDataset(
            rows=train_rows,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            judge=judge,
            convo_prefix=self.convo_prefix,
            answer_format_instruction=self.answer_format_instruction,
            dataset_name=self.dataset_name,
            n_batches=self.n_batches,
        )
        eval_ds = LLMGradedDataset(
            rows=eval_rows,
            batch_size=self.batch_size,
            group_size=1,
            renderer=renderer,
            judge=judge,
            convo_prefix=self.convo_prefix,
            answer_format_instruction=self.answer_format_instruction,
            dataset_name=f"{self.dataset_name}_eval",
        )
        return train_ds, eval_ds


@chz.chz
class GpqaOpenEndedBuilder(LLMGradedDatasetBuilder):
    subset: str = "extended"
    split: str = "train"
    seed: int = 42
    dataset_name: str = "gpqa_oe"

    def _load_rows(self) -> list[dict[str, str]]:
        from tinker_cookbook.recipes.multiplayer_rl.debate.data.gpqa import (
            load_gpqa_open_ended_rows,
        )

        raw = load_gpqa_open_ended_rows(subset=self.subset, split=self.split, seed=self.seed)
        return [{"question": str(r["question"]), "answer": str(r["answer"])} for r in raw]
