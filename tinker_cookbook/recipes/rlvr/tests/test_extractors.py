"""Tests for answer extraction functions."""

import pytest

from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.recipes.rlvr.builders import _extract_gsm8k_final_answer
from tinker_cookbook.recipes.rlvr.env import extract_final_answer


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


class TestExtractFinalAnswer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("blah <final_answer>42</final_answer> done", "42"),
            ("<final_answer>  hello  </final_answer>", "hello"),
            ("<final_answer>\nline1\nline2\n</final_answer>", "line1\nline2"),
        ],
        ids=["basic", "strips_whitespace", "multiline"],
    )
    def test_extraction(self, text: str, expected: str) -> None:
        assert extract_final_answer(text) == expected

    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_final_answer("no tags here")


# ---------------------------------------------------------------------------
# extract_boxed
# ---------------------------------------------------------------------------


class TestExtractBoxed:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The answer is \\boxed{42}", "42"),
            ("\\boxed{\\frac{n}{2}}", "\\frac{n}{2}"),
            ("\\boxed{\\frac{1}{\\sqrt{2}}}", "\\frac{1}{\\sqrt{2}}"),
            ("First \\boxed{wrong}, then \\boxed{right}", "right"),
            ("\\boxed{  x + 1  }", "  x + 1  "),
            (
                "\\boxed{1 + \\left\\lceil \\frac{n}{2} \\right\\rceil}",
                "1 + \\left\\lceil \\frac{n}{2} \\right\\rceil",
            ),
            ("\\boxed{}", ""),
            (
                "Let me work through this...\n"
                "Step 1: compute \\boxed{intermediate}\n"
                "Step 2: therefore \\boxed{\\frac{3}{4}}\n",
                "\\frac{3}{4}",
            ),
        ],
        ids=[
            "basic",
            "nested_braces",
            "deeply_nested",
            "last_boxed_wins",
            "preserves_inner_whitespace",
            "latex_ceiling",
            "empty_boxed",
            "multi_step_solution",
        ],
    )
    def test_extraction(self, text: str, expected: str) -> None:
        assert extract_boxed(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["no boxed here", "\\boxed{unclosed"],
        ids=["missing", "unclosed_brace"],
    )
    def test_raises(self, text: str) -> None:
        with pytest.raises(ValueError):
            extract_boxed(text)


# ---------------------------------------------------------------------------
# _extract_gsm8k_final_answer
# ---------------------------------------------------------------------------


class TestExtractGsm8kFinalAnswer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("some work\n#### 42", "42"),
            ("work\n#### : 42", "42"),
            ("work\n#### 1,234", "1234"),
            ("step 1\n#### 10\nstep 2\n#### 20", "20"),
            ("####   99   ", "99"),
            ("#### 1,000,000", "1000000"),
            ("#### -5", "-5"),
        ],
        ids=[
            "standard",
            "colon_after_hashes",
            "comma_stripping",
            "multiple_takes_last",
            "whitespace_around_answer",
            "large_comma_number",
            "negative_number",
        ],
    )
    def test_extraction(self, text: str, expected: str) -> None:
        assert _extract_gsm8k_final_answer(text) == expected

    def test_no_hashes_raises(self) -> None:
        with pytest.raises(ValueError, match="No GSM8K final answer"):
            _extract_gsm8k_final_answer("just some text without markers")
