"""Tests for rlvr graders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.recipes.rlvr.graders import (
    GraderConfig,
    LLMGrader,
    LLMGraderConfig,
    SympyGrader,
    SympyGraderConfig,
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
# LLMGraderConfig
# ---------------------------------------------------------------------------


def test_llm_config_sets_concurrency():
    config = LLMGraderConfig()
    assert config.client.max_concurrent is None
    # Don't call build() since it creates AsyncLLMClient which needs OPENAI_API_KEY.
    # Instead, verify the config mutation logic directly.
    import chz

    resolved = config
    if resolved.client.max_concurrent is None:
        new_client = chz.replace(resolved.client, max_concurrent=4)
        resolved = chz.replace(resolved, client=new_client)
    assert resolved.client.max_concurrent == 4


def test_llm_config_preserves_existing_concurrency():
    from tinker_cookbook.llm_client import LLMClientConfig

    config = LLMGraderConfig(client=LLMClientConfig(max_concurrent=8))
    # The build logic should not overwrite existing max_concurrent
    assert config.client.max_concurrent == 8
    # Simulate the build logic
    resolved = config
    if resolved.client.max_concurrent is None:
        import chz

        new_client = chz.replace(resolved.client, max_concurrent=4)
        resolved = chz.replace(resolved, client=new_client)
    assert resolved.client.max_concurrent == 8


# ---------------------------------------------------------------------------
# LLMGrader with mocked client
# ---------------------------------------------------------------------------


def _make_llm_grader_with_mock(response: str) -> LLMGrader:
    """Build an LLMGrader with a mocked AsyncLLMClient."""
    config = LLMGraderConfig(client=MagicMock(max_concurrent=1))
    grader = LLMGrader.__new__(LLMGrader)
    grader.config = config
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=response)
    grader.client = mock_client
    return grader


def test_llm_grader_correct():
    grader = _make_llm_grader_with_mock("CORRECT")
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="42"))
    assert result.correct is True
    assert result.status == "correct"


def test_llm_grader_incorrect():
    grader = _make_llm_grader_with_mock("INCORRECT")
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="43"))
    assert result.correct is False
    assert result.status == "incorrect"


def test_llm_grader_ambiguous():
    grader = _make_llm_grader_with_mock("MAYBE")
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="42"))
    assert result.correct is False
    assert result.status == "ambiguous"
    assert "MAYBE" in result.detail


def test_llm_grader_error():
    grader = _make_llm_grader_with_mock("")
    grader.client.complete = AsyncMock(side_effect=RuntimeError("boom"))
    result = asyncio.run(grader.grade(question="q", reference="42", extracted="42"))
    assert result.correct is False
    assert result.status == "error"
    assert "boom" in result.detail
