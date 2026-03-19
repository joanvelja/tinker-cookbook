"""Tests for rlvr graders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.llm_client import AsyncLLMClient, LLMClientConfig
from tinker_cookbook.recipes.rlvr.graders import (
    CompositeGrader,
    CompositeGraderConfig,
    LLMGrader,
    LLMGraderConfig,
    SympyGrader,
    SympyGraderConfig,
    _extract_tag,
    _llm_grade,
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


@pytest.mark.parametrize("raw", ["MAYBE", "yes", ""])
def test_parse_verdict_ambiguous(raw: str):
    result = _parse_verdict(raw)
    assert result.correct is False
    assert result.status == "ambiguous"


def test_parse_verdict_multiline():
    """Multi-line grader output: last whole-word match wins."""
    r = _parse_verdict("Core of gold: X\nCore of model: X\nVerdict: CORRECT")
    assert r.correct is True
    r = _parse_verdict("Core of gold: X\nCore of model: Y\nVerdict: INCORRECT")
    assert r.correct is False


def test_parse_verdict_reasoning_then_verdict():
    """Reasoning before verdict: last keyword wins."""
    r = _parse_verdict("The answers differ. INCORRECT")
    assert r.correct is False
    r = _parse_verdict("Initially seems wrong but actually CORRECT")
    assert r.correct is True


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
    # LLMGrader has a cache dict initialized in __init__; set it here since
    # we bypass __init__ via __new__.
    if cls is LLMGrader:
        grader._cache = {}
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

    def test_batch_skips_batch_path_with_decision_tag(self):
        """When decision_tag is set, batch goes straight to individual calls."""
        grader = _make_llm_grader("CORRECT")
        grader.config = MagicMock(
            system_prompt="sys", user_template="Target: {target}\nResponse: {response}",
            decision_tag="verdict",
        )
        results = asyncio.run(grader.grade_batch([("q", "1", "1"), ("q", "2", "2")]))
        assert len(results) == 2
        # Each request gets its own call (no batch attempt)
        assert grader.client.complete.call_count == 2


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


# ---------------------------------------------------------------------------
# _extract_tag
# ---------------------------------------------------------------------------


class TestExtractTag:
    def test_tag_present(self):
        assert _extract_tag("<verdict>CORRECT</verdict>", "verdict") == "CORRECT"

    def test_tag_with_whitespace(self):
        assert _extract_tag("<verdict>  INCORRECT  </verdict>", "verdict") == "INCORRECT"

    def test_tag_absent(self):
        assert _extract_tag("No tags here", "verdict") is None

    def test_tag_malformed_no_close(self):
        assert _extract_tag("<verdict>CORRECT", "verdict") is None

    def test_tag_malformed_wrong_name(self):
        assert _extract_tag("<answer>CORRECT</answer>", "verdict") is None

    def test_tag_multiline_content(self):
        text = "<verdict>\nThe answer is\nCORRECT\n</verdict>"
        assert _extract_tag(text, "verdict") == "The answer is\nCORRECT"

    def test_tag_embedded_in_reasoning(self):
        text = "Let me think... the answer matches. <verdict>CORRECT</verdict> Done."
        assert _extract_tag(text, "verdict") == "CORRECT"


# ---------------------------------------------------------------------------
# decision_tag flow in _llm_grade
# ---------------------------------------------------------------------------


class TestDecisionTagFlow:
    def _make_client(self, response: str) -> AsyncLLMClient:
        client = AsyncMock(spec=AsyncLLMClient)
        client.complete = AsyncMock(return_value=response)
        return client

    def test_tag_found_correct_truncated(self):
        """Stop sequence strips closing tag — _llm_grade re-appends it."""
        client = self._make_client("Reasoning here. <verdict>CORRECT")
        result = asyncio.run(_llm_grade(
            client, "system", "Target: {target}\nResponse: {response}",
            "q", "42", "42", decision_tag="verdict",
        ))
        assert result.correct is True
        assert result.status == "correct"

    def test_tag_found_correct_with_closing(self):
        """Also works when closing tag is present (no stop sequence)."""
        client = self._make_client("Reasoning here. <verdict>CORRECT</verdict>")
        result = asyncio.run(_llm_grade(
            client, "system", "Target: {target}\nResponse: {response}",
            "q", "42", "42", decision_tag="verdict",
        ))
        assert result.correct is True
        assert result.status == "correct"

    def test_tag_found_incorrect_truncated(self):
        client = self._make_client("<verdict>INCORRECT")
        result = asyncio.run(_llm_grade(
            client, "system", "Target: {target}\nResponse: {response}",
            "q", "42", "43", decision_tag="verdict",
        ))
        assert result.correct is False
        assert result.status == "incorrect"

    def test_tag_missing_returns_error(self):
        """No tag found => status='error', no regex fallback."""
        client = self._make_client("CORRECT")  # raw text has verdict but no tag
        result = asyncio.run(_llm_grade(
            client, "system", "Target: {target}\nResponse: {response}",
            "q", "42", "42", decision_tag="verdict",
        ))
        assert result.correct is False
        assert result.status == "error"
        assert "No <verdict> tag found" in result.detail

    def test_tag_with_ambiguous_content_truncated(self):
        """Tag present but content is ambiguous (stop-truncated)."""
        client = self._make_client("<verdict>MAYBE")
        result = asyncio.run(_llm_grade(
            client, "system", "Target: {target}\nResponse: {response}",
            "q", "42", "42", decision_tag="verdict",
        ))
        assert result.correct is False
        assert result.status == "ambiguous"

    def test_no_decision_tag_uses_raw_parse(self):
        """When decision_tag is None, fall through to raw _parse_verdict."""
        client = self._make_client("CORRECT")
        result = asyncio.run(_llm_grade(
            client, "system", "Target: {target}\nResponse: {response}",
            "q", "42", "42", decision_tag=None,
        ))
        assert result.correct is True
        assert result.status == "correct"


# ---------------------------------------------------------------------------
# LLMGrader result cache
# ---------------------------------------------------------------------------


class TestLLMGraderCache:
    def test_cache_hit(self):
        grader = _make_llm_grader("CORRECT")
        r1 = asyncio.run(grader.grade("q", "42", "42"))
        r2 = asyncio.run(grader.grade("q", "42", "42"))
        assert r1 is r2  # same object from cache
        grader.client.complete.assert_called_once()  # only one API call

    def test_cache_miss_different_inputs(self):
        grader = _make_llm_grader("CORRECT")
        asyncio.run(grader.grade("q1", "42", "42"))
        asyncio.run(grader.grade("q2", "42", "42"))
        assert grader.client.complete.call_count == 2

    def test_cache_stores_errors(self):
        """Errors are cached too — don't retry on the same input."""
        grader = _make_llm_grader("")
        grader.client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        r1 = asyncio.run(grader.grade("q", "42", "42"))
        r2 = asyncio.run(grader.grade("q", "42", "42"))
        assert r1.status == "error"
        assert r1 is r2
        grader.client.complete.assert_called_once()


# ---------------------------------------------------------------------------
# LLMClientConfig: max_tokens and stop wiring
# ---------------------------------------------------------------------------


class TestLLMClientConfigWiring:
    @patch("tinker_cookbook.llm_client.AsyncOpenAI")
    def test_max_tokens_passed(self, mock_openai_cls):
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_openai

        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            cfg = LLMClientConfig(max_concurrent=1, max_tokens=100)
            client = AsyncLLMClient(cfg)
            asyncio.run(client.complete(system="sys", user="usr"))

        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 100

    @patch("tinker_cookbook.llm_client.AsyncOpenAI")
    def test_stop_passed(self, mock_openai_cls):
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_openai

        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            cfg = LLMClientConfig(max_concurrent=1, stop=["</verdict>"])
            client = AsyncLLMClient(cfg)
            asyncio.run(client.complete(system="sys", user="usr"))

        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("stop") == ["</verdict>"]

    @patch("tinker_cookbook.llm_client.AsyncOpenAI")
    def test_max_tokens_and_stop_omitted_when_none(self, mock_openai_cls):
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_openai

        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            cfg = LLMClientConfig(max_concurrent=1)
            client = AsyncLLMClient(cfg)
            asyncio.run(client.complete(system="sys", user="usr"))

        call_kwargs = mock_openai.chat.completions.create.call_args
        assert "max_tokens" not in call_kwargs.kwargs
        assert "stop" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# LLMGraderConfig.build() with decision_tag
# ---------------------------------------------------------------------------


class TestLLMGraderConfigBuild:
    @patch("tinker_cookbook.recipes.rlvr.graders.AsyncLLMClient")
    def test_build_appends_format_instruction(self, mock_client_cls):
        config = LLMGraderConfig(
            client=LLMClientConfig(max_concurrent=1),
            decision_tag="verdict",
        )
        grader = config.build()
        assert "<verdict>CORRECT</verdict>" in grader.config.system_prompt
        assert "<verdict>INCORRECT</verdict>" in grader.config.system_prompt

    @patch("tinker_cookbook.recipes.rlvr.graders.AsyncLLMClient")
    def test_build_sets_stop_sequence(self, mock_client_cls):
        config = LLMGraderConfig(
            client=LLMClientConfig(max_concurrent=1),
            decision_tag="verdict",
        )
        grader = config.build()
        assert grader.config.client.stop == ["</verdict>"]

    @patch("tinker_cookbook.recipes.rlvr.graders.AsyncLLMClient")
    def test_build_no_tag_leaves_prompt_unchanged(self, mock_client_cls):
        original_prompt = "Grade: CORRECT or INCORRECT."
        config = LLMGraderConfig(
            client=LLMClientConfig(max_concurrent=1),
            system_prompt=original_prompt,
            decision_tag=None,
        )
        grader = config.build()
        assert grader.config.system_prompt == original_prompt
        assert grader.config.client.stop is None
