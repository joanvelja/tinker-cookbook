"""Tests for rlvr graders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.recipes.rlvr.graders import (
    CompositeGrader,
    CompositeGraderConfig,
    LLMGrader,
    LLMGraderConfig,
    SympyGrader,
    SympyGraderConfig,
    _parse_verdict,
)
from tinker_cookbook.recipes.rlvr.types import GradeResult


# ---------------------------------------------------------------------------
# GradeResult basics
# ---------------------------------------------------------------------------


def test_grade_result_construction():
    r = GradeResult(correct=True, status="correct", detail="ok")
    assert r.correct is True
    assert r.status == "correct"
    assert r.detail == "ok"


def test_grade_result_default_detail():
    r = GradeResult(correct=False, status="incorrect")
    assert r.detail == ""


def test_grade_result_frozen():
    r = GradeResult(correct=True, status="correct")
    with pytest.raises(AttributeError):
        r.correct = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _parse_verdict (shared helper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_correct, expected_status",
    [
        ("CORRECT", True, "correct"),
        ("  correct  ", True, "correct"),
        ("INCORRECT", False, "incorrect"),
        (" incorrect\n", False, "incorrect"),
    ],
)
def test_parse_verdict_valid(raw: str, expected_correct: bool, expected_status: str):
    result = _parse_verdict(raw)
    assert result.correct is expected_correct
    assert result.status == expected_status


@pytest.mark.parametrize("raw", ["MAYBE", "yes", "", "CORRECT INCORRECT"])
def test_parse_verdict_ambiguous(raw: str):
    result = _parse_verdict(raw)
    assert result.correct is False
    assert result.status == "ambiguous"


# ---------------------------------------------------------------------------
# SympyGraderConfig / SympyGrader
# ---------------------------------------------------------------------------


def test_sympy_config_builds_sympy_grader():
    grader = SympyGraderConfig().build()
    assert isinstance(grader, SympyGrader)


def test_sympy_grader_correct():
    grader = SympyGrader()
    result = asyncio.run(grader.grade(question="What is 6*7?", reference="42", extracted="42"))
    assert result.correct is True
    assert result.status == "correct"


def test_sympy_grader_incorrect():
    grader = SympyGrader()
    result = asyncio.run(grader.grade(question="What is 6*7?", reference="42", extracted="43"))
    assert result.correct is False
    assert result.status == "incorrect"


# ---------------------------------------------------------------------------
# LLMGraderConfig concurrency
# ---------------------------------------------------------------------------


def test_llm_config_sets_concurrency():
    import chz

    config = LLMGraderConfig()
    assert config.client.max_concurrent is None
    new_client = chz.replace(config.client, max_concurrent=4)
    resolved = chz.replace(config, client=new_client)
    assert resolved.client.max_concurrent == 4


def test_llm_config_preserves_existing_concurrency():
    from tinker_cookbook.llm_client import LLMClientConfig

    config = LLMGraderConfig(client=LLMClientConfig(max_concurrent=8))
    # build logic should not overwrite existing max_concurrent
    assert config.client.max_concurrent == 8


# ---------------------------------------------------------------------------
# LLMGrader with mocked client
# ---------------------------------------------------------------------------


def _make_mock_grader(cls, config_cls, client_attr: str, response: str):
    """Build a grader instance with a mocked AsyncLLMClient."""
    config = config_cls(**{
        ("client" if cls is LLMGrader else "llm"): MagicMock(max_concurrent=1),
    })
    grader = cls.__new__(cls)
    grader.config = config
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=response)
    setattr(grader, client_attr, mock_client)
    return grader


def _make_llm_grader(response: str = "CORRECT") -> LLMGrader:
    return _make_mock_grader(LLMGrader, LLMGraderConfig, "client", response)


def _make_composite_grader(response: str = "CORRECT") -> CompositeGrader:
    return _make_mock_grader(CompositeGrader, CompositeGraderConfig, "llm_client", response)


@pytest.mark.parametrize(
    "response, expected_correct, expected_status",
    [
        ("CORRECT", True, "correct"),
        ("INCORRECT", False, "incorrect"),
    ],
)
def test_llm_grader_verdict(response: str, expected_correct: bool, expected_status: str):
    grader = _make_llm_grader(response)
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="42"))
    assert result.correct is expected_correct
    assert result.status == expected_status


def test_llm_grader_ambiguous():
    grader = _make_llm_grader("MAYBE")
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="42"))
    assert result.correct is False
    assert result.status == "ambiguous"
    assert "MAYBE" in result.detail


def test_llm_grader_error():
    grader = _make_llm_grader("")
    grader.client.complete = AsyncMock(side_effect=RuntimeError("boom"))
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="42"))
    assert result.correct is False
    assert result.status == "error"
    assert "boom" in result.detail


# ---------------------------------------------------------------------------
# LLMGrader.grade_batch
# ---------------------------------------------------------------------------


class TestGradeBatch:
    def test_empty_batch(self):
        results = asyncio.run(_make_llm_grader("").grade_batch([]))
        assert results == []

    def test_single_delegates_to_grade(self):
        grader = _make_llm_grader("CORRECT")
        results = asyncio.run(grader.grade_batch([("q", "42", "42")]))
        assert len(results) == 1
        assert results[0].correct is True
        grader.client.complete.assert_called_once()

    @pytest.mark.parametrize(
        "raw, expected_n, expected_correctness",
        [
            ("CORRECT\nINCORRECT\nCORRECT", 3, [True, False, True]),
            ("1. CORRECT\n2. INCORRECT\n3. CORRECT", 3, [True, False, True]),
            ("1: CORRECT\n2: INCORRECT", 2, [True, False]),
            ("1) CORRECT\n2) INCORRECT", 2, [True, False]),
        ],
    )
    def test_parse_batch_formats(
        self, raw: str, expected_n: int, expected_correctness: list[bool],
    ):
        parsed = LLMGrader._parse_batch_response(raw, expected_n)
        assert parsed is not None
        assert [r.correct for r in parsed] == expected_correctness

    def test_parse_batch_wrong_line_count(self):
        assert LLMGrader._parse_batch_response("CORRECT\nINCORRECT", expected_n=3) is None

    def test_parse_batch_ambiguous_line(self):
        assert LLMGrader._parse_batch_response("CORRECT\nMAYBE\nINCORRECT", expected_n=3) is None

    def test_batch_fallback_on_parse_failure(self):
        """When batch parse fails, grade_batch falls back to individual calls."""
        grader = _make_llm_grader("")
        grader.client.complete = AsyncMock(
            side_effect=["UNPARSEABLE BATCH\nGARBAGE", "CORRECT", "CORRECT"],
        )
        results = asyncio.run(grader.grade_batch([("q", "1", "1"), ("q", "2", "2")]))
        assert len(results) == 2
        assert all(r.correct for r in results)
        assert grader.client.complete.call_count == 3


# ---------------------------------------------------------------------------
# CompositeGrader
# ---------------------------------------------------------------------------


class TestCompositeGraderConfig:
    def test_build_returns_composite_grader(self):
        with patch("tinker_cookbook.recipes.rlvr.graders.AsyncLLMClient"):
            grader = CompositeGraderConfig(llm=MagicMock(max_concurrent=4)).build()
            assert isinstance(grader, CompositeGrader)


class TestCompositeGrader:
    def test_sympy_true_skips_llm(self):
        grader = _make_composite_grader("CORRECT")
        with patch.object(grader, "_sympy_grade", return_value=True):
            result = asyncio.run(grader.grade("q", "42", "42"))
        assert result.correct is True
        assert result.detail == "sympy"
        grader.llm_client.complete.assert_not_called()

    @pytest.mark.parametrize(
        "llm_response, expected_correct, expected_status",
        [
            ("CORRECT", True, "correct"),
            ("INCORRECT", False, "incorrect"),
            ("DUNNO", False, "ambiguous"),
        ],
    )
    def test_sympy_none_falls_through_to_llm(
        self, llm_response: str, expected_correct: bool, expected_status: str,
    ):
        grader = _make_composite_grader(llm_response)
        with patch.object(grader, "_sympy_grade", return_value=None):
            result = asyncio.run(grader.grade("q", "42", "42"))
        assert result.correct is expected_correct
        assert result.status == expected_status
        grader.llm_client.complete.assert_called_once()

    def test_sympy_none_llm_error(self):
        grader = _make_composite_grader("")
        grader.llm_client.complete = AsyncMock(side_effect=RuntimeError("api down"))
        with patch.object(grader, "_sympy_grade", return_value=None):
            result = asyncio.run(grader.grade("q", "42", "42"))
        assert result.correct is False
        assert result.status == "error"
        assert "api down" in result.detail
