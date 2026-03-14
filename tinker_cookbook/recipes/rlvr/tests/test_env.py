"""Tests for tinker_cookbook.recipes.rlvr.env."""

import asyncio

import pytest
import tinker

from tinker_cookbook.recipes.rlvr.env import (
    ANSWER_FORMAT_INSTRUCTION,
    RLVRDataset,
    RLVREnv,
    extract_final_answer,
)
from tinker_cookbook.recipes.rlvr.types import ExtractFn, GradeResult, RLVRExample
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.renderers.base import Message


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockGrader:
    """Grader that returns a fixed verdict."""

    def __init__(self, correct: bool = True, status: str = "correct"):
        self.correct = correct
        self.status = status
        self.called = False

    async def grade(self, question: str, reference: str, extracted: str) -> GradeResult:
        self.called = True
        return GradeResult(correct=self.correct, status=self.status)


class _MockTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class MockRenderer:
    """Minimal renderer: each char = one token (ord value)."""

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
    """Renderer whose parse_response always signals parse failure."""

    def parse_response(self, tokens: list[int]) -> tuple[Message, bool]:
        text = "".join(chr(t) for t in tokens)
        return Message(role="assistant", content=text), False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[int]:
    """Encode text as MockRenderer tokens (ord per char)."""
    return [ord(c) for c in text]


def _make_env(
    grader: MockGrader | None = None,
    extract_fn: ExtractFn = extract_final_answer,
    renderer: MockRenderer | None = None,
    format_coef: float = 0.1,
) -> RLVREnv:
    return RLVREnv(
        question="What is 2+2?",
        reference="4",
        renderer=renderer or MockRenderer(),
        grader=grader or MockGrader(correct=True, status="correct"),
        extract_fn=extract_fn,
        format_instruction=ANSWER_FORMAT_INSTRUCTION,
        format_coef=format_coef,
    )


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


def test_extract_final_answer_basic():
    assert extract_final_answer("blah <final_answer>42</final_answer> blah") == "42"


def test_extract_final_answer_strips():
    assert extract_final_answer("<final_answer>  hello  </final_answer>") == "hello"


def test_extract_final_answer_missing():
    with pytest.raises(ValueError, match="No <final_answer>"):
        extract_final_answer("no tags here")


# ---------------------------------------------------------------------------
# RLVREnv.check_format / check_answer
# ---------------------------------------------------------------------------


def test_rlvr_env_check_format_works():
    env = _make_env()
    assert env.check_format("some <final_answer>42</final_answer> text") is True
    assert env.check_format("no tags at all") is False


def test_rlvr_env_check_answer_works():
    env_correct = _make_env(grader=MockGrader(correct=True, status="correct"))
    assert _run(env_correct.check_answer("<final_answer>4</final_answer>")) is True

    env_wrong = _make_env(grader=MockGrader(correct=False, status="incorrect"))
    assert _run(env_wrong.check_answer("<final_answer>5</final_answer>")) is False

    # Extraction failure -> False without calling grader
    grader = MockGrader(correct=True)
    env_nofmt = _make_env(grader=grader)
    assert _run(env_nofmt.check_answer("no tags")) is False
    assert grader.called is False


# ---------------------------------------------------------------------------
# RLVREnv.step
# ---------------------------------------------------------------------------


def test_rlvr_env_step_correct():
    grader = MockGrader(correct=True, status="correct")
    env = _make_env(grader=grader, format_coef=0.1)
    content = "<final_answer>4</final_answer>"
    result = _run(env.step(_tokens(content)))

    # correct_format=1.0 (parse_success=True), correct_answer=1.0
    # reward = 0.1*(1-1) + 1.0 = 1.0
    assert result.reward == pytest.approx(1.0)
    assert result.episode_done is True
    assert result.metrics["correct"] == 1.0
    assert result.metrics["format"] == 1.0
    assert result.logs["grade_status"] == "correct"
    assert grader.called is True


def test_rlvr_env_step_incorrect():
    grader = MockGrader(correct=False, status="incorrect")
    env = _make_env(grader=grader, format_coef=0.1)
    content = "<final_answer>5</final_answer>"
    result = _run(env.step(_tokens(content)))

    # correct_format=1.0, correct_answer=0.0
    # reward = 0.1*(1-1) + 0.0 = 0.0
    assert result.reward == pytest.approx(0.0)
    assert result.metrics["correct"] == 0.0
    assert result.metrics["format"] == 1.0
    assert result.logs["grade_status"] == "incorrect"


def test_rlvr_env_step_extraction_fails():
    grader = MockGrader(correct=True, status="correct")
    env = _make_env(grader=grader, format_coef=0.1)
    content = "no tags here at all"
    result = _run(env.step(_tokens(content)))

    # Extraction failed: correct_format=0.0, correct_answer=0.0
    # reward = 0.1*(0-1) + 0.0 = -0.1
    assert result.reward == pytest.approx(-0.1)
    assert result.metrics["correct"] == 0.0
    assert result.metrics["format"] == 0.0
    assert result.logs["grade_status"] == "error"
    # Grader should NOT have been called
    assert grader.called is False


def test_rlvr_env_step_parse_fail_but_extraction_ok():
    """When renderer parse_success=False but extraction succeeds,
    correct_format should be 0.0 (from parse_success), answer still graded."""
    grader = MockGrader(correct=True, status="correct")
    env = _make_env(grader=grader, renderer=FailParseRenderer(), format_coef=0.1)
    content = "<final_answer>4</final_answer>"
    result = _run(env.step(_tokens(content)))

    # correct_format = float(False) = 0.0
    # reward = 0.1*(0-1) + 1.0 = 0.9
    assert result.reward == pytest.approx(0.9)
    assert result.metrics["format"] == 0.0
    assert result.metrics["correct"] == 1.0


# ---------------------------------------------------------------------------
# RLVRDataset
# ---------------------------------------------------------------------------


def _make_examples(n: int) -> list[RLVRExample]:
    return [RLVRExample(question=f"q{i}", reference=f"a{i}") for i in range(n)]


def test_rlvr_dataset_finite_mode():
    ds = RLVRDataset(
        examples=_make_examples(10),
        batch_size=3,
        group_size=2,
        renderer=MockRenderer(),
        grader=MockGrader(),
        extract_fn=extract_final_answer,
        n_batches=None,
    )
    assert len(ds) == 4  # ceil(10/3)

    # Last batch should have only 1 item (indices 9)
    last = ds.get_batch(3)
    assert len(last) == 1

    # First batch should have 3
    first = ds.get_batch(0)
    assert len(first) == 3


def test_rlvr_dataset_cycling_mode():
    ds = RLVRDataset(
        examples=_make_examples(10),
        batch_size=3,
        group_size=2,
        renderer=MockRenderer(),
        grader=MockGrader(),
        extract_fn=extract_final_answer,
        n_batches=5,
    )
    assert len(ds) == 5

    # Every batch has exactly batch_size items
    for i in range(5):
        batch = ds.get_batch(i)
        assert len(batch) == 3


def test_rlvr_dataset_get_batch_returns_group_builders():
    ds = RLVRDataset(
        examples=_make_examples(5),
        batch_size=2,
        group_size=4,
        renderer=MockRenderer(),
        grader=MockGrader(),
        extract_fn=extract_final_answer,
    )
    batch = ds.get_batch(0)
    assert len(batch) == 2
    for builder in batch:
        assert isinstance(builder, ProblemGroupBuilder)
        assert builder.num_envs == 4
        assert builder.dataset_name == "rlvr"
