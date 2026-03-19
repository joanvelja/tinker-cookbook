"""Grader configs and runtime graders for RLVR recipes."""

import asyncio
import logging
import math
import re
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

    async def grade_batch(
        self, requests: list[tuple[str, str, str]]
    ) -> list[GradeResult]:
        """Grade multiple (question, reference, extracted) tuples.

        Default implementation fans out to individual ``grade`` calls.
        """
        return list(await asyncio.gather(*(self.grade(q, r, e) for q, r, e in requests)))


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
    # When set, extract verdict from <tag>...</tag> instead of raw text.
    # The judge can reason freely; only the tag content is parsed.
    decision_tag: str | None = None

    def build(self, concurrency_hint: int = 1) -> "LLMGrader":
        config = self
        if config.client.max_concurrent is None:
            new_client = chz.replace(config.client, max_concurrent=concurrency_hint)
            config = chz.replace(config, client=new_client)
        # Auto-append format instruction and set stop sequence from decision_tag
        if config.decision_tag is not None:
            tag = config.decision_tag
            fmt_line = (
                f"\nPut your verdict in <{tag}>CORRECT</{tag}> or <{tag}>INCORRECT</{tag}> tags."
            )
            new_system = config.system_prompt + fmt_line
            new_client = chz.replace(config.client, stop=[f"</{tag}>"])
            config = chz.replace(config, system_prompt=new_system, client=new_client)
        return LLMGrader(config=config)


# ---------------------------------------------------------------------------
# Shared verdict parsing
# ---------------------------------------------------------------------------

_RE_CORRECT = re.compile(r"\bCORRECT\b")
_RE_INCORRECT = re.compile(r"\bINCORRECT\b")


def _parse_verdict(raw: str) -> GradeResult:
    """Parse CORRECT/INCORRECT verdict from raw LLM output.

    Strategy (mirrors nanodebate's _binary_judge):
    1. First word match — handles compliant single-word responses.
    2. Last whole-word occurrence — handles reasoning-then-verdict and
       multi-line prompts where the verdict appears after analysis.
    """
    text = raw.strip().upper()
    # First word match
    word = text.split()[0].strip(".,;:!?\"'*#()[]`") if text else ""
    if word == "CORRECT":
        return GradeResult(correct=True, status="correct")
    if word == "INCORRECT":
        return GradeResult(correct=False, status="incorrect")
    # Fallback: scan for last whole-word occurrence (last match wins).
    # \bCORRECT\b won't match inside INCORRECT (no word boundary after IN),
    # so no filtering needed.
    correct_hits = list(_RE_CORRECT.finditer(text))
    incorrect_hits = list(_RE_INCORRECT.finditer(text))
    if not correct_hits and not incorrect_hits:
        return GradeResult(
            correct=False, status="ambiguous",
            detail=f"Expected CORRECT or INCORRECT, got: {raw!r}",
        )
    last_correct = correct_hits[-1].start() if correct_hits else -1
    last_incorrect = incorrect_hits[-1].start() if incorrect_hits else -1
    if last_correct > last_incorrect:
        return GradeResult(correct=True, status="correct")
    return GradeResult(correct=False, status="incorrect")


def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content from <tag>...</tag>. Returns None if not found."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


async def _llm_grade(
    client: AsyncLLMClient,
    system_prompt: str,
    user_template: str,
    question: str,
    reference: str,
    extracted: str,
    *,
    decision_tag: str | None = None,
) -> GradeResult:
    """Call LLM and parse the verdict. Shared by LLMGrader and CompositeGrader."""
    user_prompt = user_template.format(
        question=question, target=reference, response=extracted,
    )
    try:
        raw = await client.complete(system=system_prompt, user=user_prompt)
        if decision_tag is not None:
            # The stop sequence strips the closing tag from the response;
            # re-append it so _extract_tag can match.
            raw = raw + f"</{decision_tag}>"
            tag_content = _extract_tag(raw, decision_tag)
            if tag_content is None:
                return GradeResult(
                    correct=False, status="error",
                    detail=f"No <{decision_tag}> tag found in: {raw!r}",
                )
            result = _parse_verdict(tag_content)
            # Preserve the full response (reasoning + verdict) in detail
            if not result.detail:
                result = GradeResult(
                    correct=result.correct, status=result.status, detail=raw,
                )
            return result
        return _parse_verdict(raw)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as e:
        return GradeResult(correct=False, status="error", detail=f"Grading failed: {e}")


# ---------------------------------------------------------------------------
# Composite grader (sympy first, LLM fallback)
# ---------------------------------------------------------------------------


@chz.chz
class CompositeGraderConfig(GraderConfig):
    """Try sympy first (deterministic). If sympy says True, trust it.
    If sympy says False, fall through to LLM for a second opinion.

    Sympy's True is always reliable (exact string match or proven math equivalence).
    Sympy's False can miss semantic equivalences (function defs, sqrt, prose).
    The LLM catches what sympy misses.
    """

    llm: LLMClientConfig = LLMClientConfig()
    system_prompt: str = _DEFAULT_SYSTEM
    user_template: str = _DEFAULT_USER
    sympy_timeout: float = 1.0
    sympy_backend: Literal["sympy", "math_verify"] = "sympy"

    def build(self, concurrency_hint: int = 1) -> "CompositeGrader":
        llm_config = self
        if llm_config.llm.max_concurrent is None:
            new_client = chz.replace(llm_config.llm, max_concurrent=concurrency_hint)
            llm_config = chz.replace(llm_config, llm=new_client)
        return CompositeGrader(config=llm_config)


class CompositeGrader:
    def __init__(self, config: CompositeGraderConfig) -> None:
        self.config = config
        self.llm_client = AsyncLLMClient(config.llm)

    def _sympy_grade(self, extracted: str, reference: str) -> bool | None:
        """Run sympy grading. Returns True (definitely equal), or None (unsure/False)."""
        from tinker_cookbook.recipes.math_rl.math_grading import (
            grade_answer,
            grade_answer_math_verify,
            run_with_timeout_signal,
        )

        if self.config.sympy_backend == "sympy":
            grader_func = grade_answer
        else:
            grader_func = grade_answer_math_verify

        try:
            out = run_with_timeout_signal(
                grader_func,
                args=(extracted, reference),
                timeout_seconds=int(math.ceil(self.config.sympy_timeout)),
            )
        except Exception:
            return None  # can't tell -> fall through to LLM

        if out is True:
            return True  # sympy says equal -> trust it
        return None  # sympy says False or timeout -> don't trust, ask LLM

    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult:
        sympy_result = self._sympy_grade(extracted, reference)
        if sympy_result is True:
            return GradeResult(correct=True, status="correct", detail="sympy")

        return await _llm_grade(
            self.llm_client, self.config.system_prompt, self.config.user_template,
            question, reference, extracted,
        )


# ---------------------------------------------------------------------------
# LLM-only grader
# ---------------------------------------------------------------------------


class LLMGrader:
    def __init__(self, config: LLMGraderConfig) -> None:
        self.config = config
        self.client = AsyncLLMClient(config.client)
        self._cache: dict[tuple[str, str, str], GradeResult] = {}

    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult:
        key = (question, reference, extracted)
        if key in self._cache:
            return self._cache[key]
        result = await _llm_grade(
            self.client, self.config.system_prompt, self.config.user_template,
            question, reference, extracted,
            decision_tag=self.config.decision_tag,
        )
        self._cache[key] = result
        return result

    async def grade_batch(
        self, requests: list[tuple[str, str, str]]
    ) -> list[GradeResult]:
        """Grade multiple answers in a single LLM call.

        Falls back to individual ``grade`` calls if batch parsing fails.
        """
        n = len(requests)
        if n == 0:
            return []
        if n == 1:
            return [await self.grade(*requests[0])]

        # When decision_tag is set, the stop sequence and tag format instruction
        # make batch prompts unusable — fall through to individual calls.
        if self.config.decision_tag is not None:
            return list(await asyncio.gather(*(self.grade(q, r, e) for q, r, e in requests)))

        # Build batched prompt
        lines = []
        for i, (_question, reference, extracted) in enumerate(requests, 1):
            lines.append(f"{i}. Target: {reference}, Response: {extracted}")
        user_prompt = (
            f"Grade each of the following {n} responses:\n\n"
            + "\n".join(lines)
            + f"\n\nRespond with exactly {n} lines, one per response, each either CORRECT or INCORRECT."
        )

        try:
            raw = await self.client.complete(
                system=self.config.system_prompt, user=user_prompt,
            )
            verdicts = self._parse_batch_response(raw, n)
            if verdicts is not None:
                return verdicts
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception:
            pass

        # Fallback: individual calls
        logger.warning("Batch grading parse failed (n=%d), falling back to individual calls", n)
        return list(await asyncio.gather(*(self.grade(q, r, e) for q, r, e in requests)))

    @staticmethod
    def _parse_batch_response(raw: str, expected_n: int) -> list[GradeResult] | None:
        """Parse N lines of CORRECT/INCORRECT. Returns None on failure."""
        lines = [line.strip().upper() for line in raw.strip().splitlines() if line.strip()]
        if len(lines) != expected_n:
            return None
        results: list[GradeResult] = []
        for line in lines:
            # Strip leading numbering like "1." or "1:"
            cleaned = line.lstrip("0123456789").lstrip(".):- ").strip()
            if cleaned == "CORRECT":
                results.append(GradeResult(correct=True, status="correct"))
            elif cleaned == "INCORRECT":
                results.append(GradeResult(correct=False, status="incorrect"))
            else:
                return None  # Any ambiguous line -> abort batch parse
        return results
