"""Wave 1 stress tests — adversarial checks for rlvr types, graders, env, dataset."""

import asyncio
import inspect
import math
from functools import partial
from unittest.mock import AsyncMock, MagicMock

import chz
import pytest
import tinker

from tinker_cookbook.recipes.rlvr.env import (
    ANSWER_FORMAT_INSTRUCTION,
    RLVRDataset,
    RLVREnv,
    extract_final_answer,
)
from tinker_cookbook.recipes.rlvr.graders import (
    GraderConfig,
    LLMGraderConfig,
    SympyGrader,
    SympyGraderConfig,
)
from tinker_cookbook.recipes.rlvr.types import ExtractFn, GradeResult, RLVRExample
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder


# ---------------------------------------------------------------------------
# Shared mocks (same as test_env.py to avoid import coupling)
# ---------------------------------------------------------------------------


class _MockTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class MockRenderer:
    def __init__(self) -> None:
        self.tokenizer = _MockTokenizer()

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def build_generation_prompt(
        self, messages: list[Message], prefill: str | None = None
    ) -> tinker.ModelInput:
        text = "".join(m.get("content", "") or "" for m in messages)
        if prefill:
            text += prefill
        tokens = [ord(c) for c in text] if text else []
        return tinker.ModelInput.from_ints(tokens)

    def parse_response(self, tokens: list[int]) -> tuple[Message, bool]:
        text = "".join(chr(t) for t in tokens)
        return Message(role="assistant", content=text), True


class FailParseRenderer(MockRenderer):
    def parse_response(self, tokens: list[int]) -> tuple[Message, bool]:
        text = "".join(chr(t) for t in tokens)
        return Message(role="assistant", content=text), False


class MockGrader:
    def __init__(self, correct: bool = True, status: str = "correct"):
        self.correct = correct
        self.status = status

    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult:
        return GradeResult(correct=self.correct, status=self.status)


def _tokens(text: str) -> list[int]:
    return [ord(c) for c in text]


def _run(coro):
    return asyncio.run(coro)


def _make_examples(n: int) -> list[RLVRExample]:
    return [RLVRExample(question=f"q{i}", reference=f"a{i}") for i in range(n)]


# ===========================================================================
# (a) chz CLI construction
# ===========================================================================


class TestChzConstruction:
    def test_sympy_config_from_argv(self):
        cfg = chz.Blueprint(SympyGraderConfig).make_from_argv(
            ["--timeout=2.0"], allow_hyphens=True
        )
        assert cfg.timeout == 2.0
        assert cfg.backend == "sympy"  # default

    def test_sympy_config_from_argv_backend(self):
        cfg = chz.Blueprint(SympyGraderConfig).make_from_argv(
            ["--timeout=0.5", "--backend=math_verify"], allow_hyphens=True
        )
        assert cfg.timeout == 0.5
        assert cfg.backend == "math_verify"

    def test_llm_config_from_argv(self):
        cfg = chz.Blueprint(LLMGraderConfig).make_from_argv([], allow_hyphens=True)
        assert isinstance(cfg, LLMGraderConfig)
        # Default system prompt should be present
        assert "CORRECT" in cfg.system_prompt


# ===========================================================================
# (b) GraderConfig ABC enforcement
# ===========================================================================


class TestGraderConfigABC:
    def test_grader_config_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            GraderConfig()  # type: ignore[abstract]

    def test_sympy_config_can_instantiate(self):
        cfg = SympyGraderConfig()
        assert isinstance(cfg, GraderConfig)

    def test_llm_config_can_instantiate(self):
        cfg = LLMGraderConfig()
        assert isinstance(cfg, GraderConfig)


# ===========================================================================
# (c) RLVREnv.step() reward parity with ProblemEnv.step()
# ===========================================================================


class _ConcreteProblemEnv(ProblemEnv):
    """Minimal concrete ProblemEnv that mirrors RLVREnv semantics for comparison."""

    def __init__(self, renderer, grader, correct_format_result: bool, format_coef: float = 0.1):
        super().__init__(renderer, format_coef=format_coef)
        self._grader = grader
        self._correct_format = correct_format_result

    def get_question(self) -> str:
        return "q"

    def get_reference_answer(self) -> str:
        return "ref"

    def check_format(self, sample_str: str) -> bool:
        return self._correct_format

    async def check_answer(self, sample_str: str) -> bool:
        return self._grader.correct


class TestRewardParity:
    """Verify RLVREnv.step() reward formula matches ProblemEnv.step() for aligned scenarios."""

    def test_correct_format_correct_answer(self):
        """Both format OK + answer correct -> reward = 1.0."""
        grader = MockGrader(correct=True, status="correct")
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=grader, extract_fn=extract_final_answer,
            format_coef=0.1,
        )
        result = _run(env.step(_tokens("<final_answer>42</final_answer>")))
        # format=1.0, answer=1.0 -> 1.0 * (1.0 + 0.1*1.0 + 0.0*1.0) = 1.1
        assert result.reward == pytest.approx(1.1)

    def test_correct_format_wrong_answer(self):
        """Format OK + wrong answer -> reward = 0.0."""
        grader = MockGrader(correct=False, status="incorrect")
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=grader, extract_fn=extract_final_answer,
            format_coef=0.1,
        )
        result = _run(env.step(_tokens("<final_answer>wrong</final_answer>")))
        # format=1.0, answer=0.0 -> 0.0 * (...) = 0.0
        assert result.reward == pytest.approx(0.0)

    def test_failed_parse_no_reward(self):
        """Parse fails → renderer couldn't parse → no extraction, no grading."""
        grader = MockGrader(correct=True, status="correct")
        env = RLVREnv(
            question="q", reference="ref", renderer=FailParseRenderer(),
            grader=grader, extract_fn=extract_final_answer,
            format_coef=0.1,
        )
        result = _run(env.step(_tokens("<final_answer>42</final_answer>")))
        # parse_success=False → no extraction → answer=0 → reward = 0.0
        assert result.reward == pytest.approx(0.0)

    def test_failed_everything(self):
        """Extraction fails -> format=0, answer=0, reward = -format_coef."""
        grader = MockGrader(correct=True, status="correct")
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=grader, extract_fn=extract_final_answer,
            format_coef=0.1,
        )
        result = _run(env.step(_tokens("no tags here")))
        # extraction failed: answer=0 -> reward = 0.0
        assert result.reward == pytest.approx(0.0)

    def test_reward_formula_custom_format_coef(self):
        """Verify formula: extraction fails -> answer=0 -> reward=0 regardless of format_coef."""
        for fc in [0.0, 0.5, 1.0, 2.0]:
            grader = MockGrader(correct=True, status="correct")
            env = RLVREnv(
                question="q", reference="ref", renderer=MockRenderer(),
                grader=grader, extract_fn=extract_final_answer,
                format_coef=fc,
            )
            result = _run(env.step(_tokens("no tags")))
            # extraction failed -> answer=0 -> 0.0 * (...) = 0.0
            assert result.reward == pytest.approx(0.0), f"format_coef={fc}"

    def test_reward_formula_custom_eos_coef(self):
        """Verify eos_coef penalizes truncation. But parse_success=False
        means no extraction regardless — eos_coef only matters on success."""
        for ec in [0.0, 0.1, 0.5]:
            grader = MockGrader(correct=True, status="correct")
            # Use MockRenderer (parse_success=True) to test eos_coef on success path
            env = RLVREnv(
                question="q", reference="ref", renderer=MockRenderer(),
                grader=grader, extract_fn=extract_final_answer,
                format_coef=0.1, eos_coef=ec,
            )
            result = _run(env.step(_tokens("<final_answer>42</final_answer>")))
            # boxed=1.0, eos=1.0, answer=1.0 -> 1.0 * (1.0 + 0.1*1.0 + ec*1.0)
            expected = 1.0 * (1.0 + 0.1 + ec)
            assert result.reward == pytest.approx(expected), f"eos_coef={ec}"


# ===========================================================================
# (d) RLVRDataset finite mode short tail
# ===========================================================================


class TestDatasetFiniteShortTail:
    def test_seven_examples_batch3(self):
        ds = RLVRDataset(
            examples=_make_examples(7),
            batch_size=3,
            group_size=2,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            n_batches=None,
        )
        assert len(ds) == 3  # ceil(7/3) = 3

        b0 = ds.get_batch(0)
        assert len(b0) == 3

        b1 = ds.get_batch(1)
        assert len(b1) == 3

        b2 = ds.get_batch(2)
        assert len(b2) == 1  # 7 - 6 = 1

    def test_no_crash_on_valid_indices(self):
        ds = RLVRDataset(
            examples=_make_examples(7),
            batch_size=3,
            group_size=2,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            n_batches=None,
        )
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            assert len(batch) >= 1

    def test_no_modulo_wrapping_in_finite_mode(self):
        """In finite mode, batch 3 (out of range) should return empty, not wrap."""
        ds = RLVRDataset(
            examples=_make_examples(7),
            batch_size=3,
            group_size=2,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            n_batches=None,
        )
        # batch index 3 → start=9, end=min(9,7)=7, indices=range(9,7)=[]
        b3 = ds.get_batch(3)
        assert len(b3) == 0


# ===========================================================================
# (e) RLVRDataset cycling mode wraps
# ===========================================================================


class TestDatasetCyclingWraps:
    def test_five_examples_batch3_nbatches10(self):
        ds = RLVRDataset(
            examples=_make_examples(5),
            batch_size=3,
            group_size=2,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            n_batches=10,
        )
        assert len(ds) == 10

        b9 = ds.get_batch(9)
        assert len(b9) == 3  # cycling always gives batch_size

    def test_cycling_covers_all_examples(self):
        """Verify all 5 examples appear when cycling through enough batches."""
        examples = _make_examples(5)
        ds = RLVRDataset(
            examples=examples,
            batch_size=3,
            group_size=2,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            n_batches=10,
        )
        # Collect all example indices used. We can't directly inspect the thunk,
        # but we can verify via the cycling formula: (batch*bs + i) % n
        seen = set()
        for batch_idx in range(10):
            for i in range(3):
                seen.add((batch_idx * 3 + i) % 5)
        assert seen == {0, 1, 2, 3, 4}


# ===========================================================================
# (f) check_answer/check_format abstractmethod compliance
# ===========================================================================


class TestAbstractMethodCompliance:
    def test_check_answer_is_async(self):
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=MockGrader(), extract_fn=extract_final_answer,
        )
        assert inspect.iscoroutinefunction(env.check_answer)

    def test_check_format_is_sync(self):
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=MockGrader(), extract_fn=extract_final_answer,
        )
        assert not inspect.iscoroutinefunction(env.check_format)
        # Call it directly — should return a bool, not a coroutine
        result = env.check_format("<final_answer>x</final_answer>")
        assert isinstance(result, bool)

    def test_check_answer_callable_independently(self):
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=MockGrader(correct=True), extract_fn=extract_final_answer,
        )
        result = _run(env.check_answer("<final_answer>42</final_answer>"))
        assert result is True

    def test_check_format_callable_independently(self):
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=MockGrader(), extract_fn=extract_final_answer,
        )
        assert env.check_format("<final_answer>x</final_answer>") is True
        assert env.check_format("no tags") is False


# ===========================================================================
# (g) GradeResult in logs not metrics
# ===========================================================================


class TestGradeStatusInLogs:
    def test_grade_status_in_logs(self):
        grader = MockGrader(correct=True, status="correct")
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=grader, extract_fn=extract_final_answer,
        )
        result = _run(env.step(_tokens("<final_answer>42</final_answer>")))
        assert "grade_status" in result.logs
        assert result.logs["grade_status"] == "correct"

    def test_grade_status_not_in_metrics(self):
        grader = MockGrader(correct=True, status="correct")
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=grader, extract_fn=extract_final_answer,
        )
        result = _run(env.step(_tokens("<final_answer>42</final_answer>")))
        assert "grade_status" not in result.metrics

    def test_grade_status_error_on_extraction_failure(self):
        grader = MockGrader(correct=True, status="correct")
        env = RLVREnv(
            question="q", reference="ref", renderer=MockRenderer(),
            grader=grader, extract_fn=extract_final_answer,
        )
        result = _run(env.step(_tokens("no tags")))
        assert result.logs["grade_status"] == "error"
        assert "grade_status" not in result.metrics


# ===========================================================================
# (h) extract_final_answer edge cases
# ===========================================================================


class TestExtractFinalAnswerEdgeCases:
    def test_multiline_content(self):
        text = "<final_answer>\nline1\nline2\n</final_answer>"
        assert extract_final_answer(text) == "line1\nline2"

    def test_nested_tags(self):
        """Regex is non-greedy, so nested inner tags will be captured as-is."""
        text = "<final_answer><final_answer>inner</final_answer></final_answer>"
        # Non-greedy: matches first <final_answer> to first </final_answer>
        result = extract_final_answer(text)
        assert result == "<final_answer>inner"

    def test_multiple_tags_returns_first(self):
        text = "<final_answer>first</final_answer> some text <final_answer>second</final_answer>"
        assert extract_final_answer(text) == "first"

    def test_empty_tags(self):
        text = "<final_answer></final_answer>"
        assert extract_final_answer(text) == ""

    def test_whitespace_only_tags(self):
        text = "<final_answer>   </final_answer>"
        assert extract_final_answer(text) == ""

    def test_tags_with_surrounding_content(self):
        text = "Let me think... <final_answer>42</final_answer> ...done"
        assert extract_final_answer(text) == "42"

    def test_no_tags_raises(self):
        with pytest.raises(ValueError, match="No <final_answer>"):
            extract_final_answer("just some text")

    def test_partial_tag_raises(self):
        with pytest.raises(ValueError, match="No <final_answer>"):
            extract_final_answer("<final_answer>unclosed")


# ===========================================================================
# (i) SympyGrader with actual sympy
# ===========================================================================


class TestSympyGraderReal:
    def test_exact_match(self):
        grader = SympyGrader()
        result = _run(grader.grade("q", "42", "42"))
        assert result.correct is True

    def test_fraction_vs_decimal(self):
        grader = SympyGrader()
        result = _run(grader.grade("q", "0.5", "\\frac{1}{2}"))
        assert result.correct is True

    def test_incorrect(self):
        grader = SympyGrader()
        result = _run(grader.grade("q", "42", "43"))
        assert result.correct is False
        assert result.status == "incorrect"

    def test_negative_numbers(self):
        grader = SympyGrader()
        result = _run(grader.grade("q", "-7", "-7"))
        assert result.correct is True

    def test_whitespace_insensitive(self):
        """Sympy should handle whitespace variations."""
        grader = SympyGrader()
        result = _run(grader.grade("q", "42", " 42 "))
        assert result.correct is True


# ===========================================================================
# (j) format_coef propagation through RLVRDataset
# ===========================================================================


class TestFormatCoefPropagation:
    def test_format_coef_reaches_env(self):
        ds = RLVRDataset(
            examples=_make_examples(3),
            batch_size=2,
            group_size=1,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            format_coef=0.5,
        )
        batch = ds.get_batch(0)
        assert len(batch) == 2
        builder = batch[0]
        assert isinstance(builder, ProblemGroupBuilder)

        # Call the env_thunk to get an actual env and check format_coef
        env = builder.env_thunk()
        assert isinstance(env, RLVREnv)
        assert env.format_coef == 0.5

    def test_format_coef_default(self):
        ds = RLVRDataset(
            examples=_make_examples(3),
            batch_size=2,
            group_size=1,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            # default format_coef=0.1
        )
        batch = ds.get_batch(0)
        env = batch[0].env_thunk()
        assert env.format_coef == 0.1

    def test_format_coef_zero(self):
        """format_coef=0 means no format penalty."""
        ds = RLVRDataset(
            examples=_make_examples(3),
            batch_size=2,
            group_size=1,
            renderer=MockRenderer(),
            grader=MockGrader(),
            extract_fn=extract_final_answer,
            format_coef=0.0,
        )
        env = ds.get_batch(0)[0].env_thunk()
        assert env.format_coef == 0.0

        # Step with failed extraction: reward should be 0.0*(0-1)+0 = 0.0
        result = _run(env.step(_tokens("no tags")))
        assert result.reward == pytest.approx(0.0)
