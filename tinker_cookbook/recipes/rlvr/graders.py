"""Grader configs and runtime graders for RLVR recipes."""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from typing import Literal, Protocol, runtime_checkable

import chz

from tinker_cookbook.llm_client import AsyncLLMClient, LLMClientConfig
from tinker_cookbook.recipes.rlvr.types import GradeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grader protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Grader(Protocol):
    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult: ...


# ---------------------------------------------------------------------------
# Abstract config
# ---------------------------------------------------------------------------


@chz.chz
class GraderConfig(ABC):
    @abstractmethod
    def build(self, concurrency_hint: int = 1) -> Grader:
        pass


# ---------------------------------------------------------------------------
# Sympy grader
# ---------------------------------------------------------------------------


@chz.chz
class SympyGraderConfig(GraderConfig):
    timeout: float = 1.0
    backend: Literal["sympy", "math_verify"] = "sympy"

    def build(self, concurrency_hint: int = 1) -> "SympyGrader":
        return SympyGrader(timeout=self.timeout, backend=self.backend)


class SympyGrader:
    def __init__(self, timeout: float = 1.0, backend: Literal["sympy", "math_verify"] = "sympy"):
        self.timeout = timeout
        self.backend = backend

    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult:
        from tinker_cookbook.recipes.math_rl.math_grading import (
            grade_answer,
            grade_answer_math_verify,
            run_with_timeout_signal,
        )

        if self.backend == "sympy":
            grader_func = grade_answer
        elif self.backend == "math_verify":
            grader_func = grade_answer_math_verify
        else:
            return GradeResult(correct=False, status="error", detail=f"Unknown backend: {self.backend}")

        try:
            out = run_with_timeout_signal(
                grader_func,
                args=(extracted, reference),
                timeout_seconds=int(math.ceil(self.timeout)),
            )
        except Exception as e:
            return GradeResult(correct=False, status="error", detail=str(e))

        if out is None:
            return GradeResult(correct=False, status="timeout", detail="Grading timed out")
        elif out:
            return GradeResult(correct=True, status="correct")
        else:
            return GradeResult(correct=False, status="incorrect")


# ---------------------------------------------------------------------------
# LLM grader
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """\
Grade: CORRECT or INCORRECT.
CORRECT = same claim (format/verbosity/notation differences are fine).
INCORRECT = different claim (even subtly: wrong number, name, sign, mechanism).
Examples:
  CORRECT: '6' vs 'Six hydrogen atoms' — same value.
  CORRECT: 'Fe2O3' vs 'iron(III) oxide' — same compound.
  INCORRECT: '0.32' vs '0.33' — different number.
  INCORRECT: '(1S,2S)' vs '(1R,2S)' — different stereochemistry.
One word."""

_DEFAULT_USER = "Target: {target}\nResponse: {response}"


@chz.chz
class LLMGraderConfig(GraderConfig):
    client: LLMClientConfig = LLMClientConfig()
    system_prompt: str = _DEFAULT_SYSTEM
    user_template: str = _DEFAULT_USER

    def build(self, concurrency_hint: int = 1) -> "LLMGrader":
        config = self
        if config.client.max_concurrent is None:
            new_client = chz.replace(config.client, max_concurrent=concurrency_hint)
            config = chz.replace(config, client=new_client)
        return LLMGrader(config=config)


class LLMGrader:
    def __init__(self, config: LLMGraderConfig) -> None:
        self.config = config
        self.client = AsyncLLMClient(config.client)

    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult:
        user_prompt = self.config.user_template.format(
            question=question, target=reference, response=extracted,
        )
        try:
            raw = await self.client.complete(system=self.config.system_prompt, user=user_prompt)
            verdict = raw.strip().upper()
            if verdict == "CORRECT":
                return GradeResult(correct=True, status="correct")
            elif verdict == "INCORRECT":
                return GradeResult(correct=False, status="incorrect")
            else:
                return GradeResult(
                    correct=False,
                    status="ambiguous",
                    detail=f"Expected CORRECT or INCORRECT, got: {raw!r}",
                )
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            return GradeResult(correct=False, status="error", detail=f"Grading failed: {e}")
