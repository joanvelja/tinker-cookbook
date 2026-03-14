"""Generic RLVR environment and dataset."""

import logging
import math
import re
import time
from functools import partial
from typing import Sequence

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.recipes.rlvr.graders import Grader
from tinker_cookbook.recipes.rlvr.types import ExtractFn, RLVRExample
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, StepResult
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

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


class RLVREnv(ProblemEnv):
    def __init__(
        self,
        question: str,
        reference: str,
        renderer: renderers.Renderer,
        grader: Grader,
        extract_fn: ExtractFn,
        format_instruction: str = "",
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        eos_coef: float = 0.0,
    ):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.question = question
        self.reference = reference
        self.grader = grader
        self.extract_fn = extract_fn
        self.format_instruction = format_instruction
        self.eos_coef = eos_coef

    def get_question(self) -> str:
        return self.question + self.format_instruction

    def get_reference_answer(self) -> str:
        return self.reference

    def check_format(self, sample_str: str) -> bool:
        try:
            self.extract_fn(sample_str)
            return True
        except ValueError:
            return False

    async def check_answer(self, sample_str: str) -> bool:
        try:
            extracted = self.extract_fn(sample_str)
        except ValueError:
            return False
        result = await self.grader.grade(self.question, self.reference, extracted)
        return result.correct

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Two independent format signals:
        #   correct_boxed: did the model use the requested answer format (e.g. \boxed{})?
        #   correct_eos:   did the response complete with a stop token (not truncated)?
        correct_eos = float(parse_success)

        if not parse_success:
            # Renderer couldn't parse → content is raw text with thinking
            # potentially embedded. Don't trust it — no extraction, no grading.
            correct_boxed = 0.0
            correct_answer = 0.0
            grade_status = "error"
            check_answer_s = 0.0
        else:
            # Renderer parsed successfully → content is clean visible text.
            try:
                extracted = self.extract_fn(content)
            except ValueError:
                correct_boxed = 0.0
                correct_answer = 0.0
                grade_status = "error"
                check_answer_s = 0.0
            else:
                correct_boxed = 1.0
                t0 = time.monotonic()
                result = await self.grader.grade(self.question, self.reference, extracted)
                check_answer_s = time.monotonic() - t0
                correct_answer = float(result.correct)
                grade_status = result.status

        # reward = format_coef * (boxed - 1) + eos_coef * (eos - 1) + correct
        total_reward = (
            self.format_coef * (correct_boxed - 1)
            + self.eos_coef * (correct_eos - 1)
            + correct_answer
        )

        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Boxed: {'\u2713' if correct_boxed else '\u2717'}, "
            f"EOS: {'\u2713' if correct_eos else '\u2717'}, "
            f"Correct: {'\u2713' if correct_answer else '\u2717'}, "
            f"Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format_boxed": correct_boxed,
                "format_eos": correct_eos,
                "correct": correct_answer,
                "time/check_answer_s": check_answer_s,
            },
            logs={
                "grade_status": grade_status,
            },
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RLVRDataset(RLDataset):
    def __init__(
        self,
        examples: list[RLVRExample],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        grader: Grader,
        extract_fn: ExtractFn,
        format_instruction: str = "",
        convo_prefix: list[renderers.Message] | None = None,
        dataset_name: str = "rlvr",
        n_batches: int | None = None,
        format_coef: float = 0.1,
        eos_coef: float = 0.0,
    ):
        self.examples = examples
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.grader = grader
        self.extract_fn = extract_fn
        self.format_instruction = format_instruction
        self.convo_prefix = convo_prefix
        self.dataset_name = dataset_name
        self._n_batches = n_batches
        self.format_coef = format_coef
        self.eos_coef = eos_coef

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        n = len(self.examples)
        if self._n_batches is not None:
            # Cycling mode: wrap around
            indices = [(index * self.batch_size + i) % n for i in range(self.batch_size)]
        else:
            # Finite mode: stop at end
            start = index * self.batch_size
            end = min(start + self.batch_size, n)
            indices = list(range(start, end))

        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    RLVREnv,
                    question=self.examples[i].question,
                    reference=self.examples[i].reference,
                    renderer=self.renderer,
                    grader=self.grader,
                    extract_fn=self.extract_fn,
                    format_instruction=self.format_instruction,
                    convo_prefix=self.convo_prefix,
                    format_coef=self.format_coef,
                    eos_coef=self.eos_coef,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for i in indices
        ]

    def __len__(self) -> int:
        if self._n_batches is not None:
            return self._n_batches
        return math.ceil(len(self.examples) / self.batch_size)
