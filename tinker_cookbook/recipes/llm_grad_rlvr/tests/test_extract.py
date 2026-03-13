"""Tests for answer extraction functions."""

import pytest

from tinker_cookbook.recipes.llm_grad_rlvr.env import extract_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


class TestExtractFinalAnswer:
    def test_basic(self):
        assert extract_final_answer("blah <final_answer>42</final_answer> done") == "42"

    def test_strips_whitespace(self):
        assert extract_final_answer("<final_answer>  hello  </final_answer>") == "hello"

    def test_missing(self):
        with pytest.raises(ValueError):
            extract_final_answer("no tags here")

    def test_multiline(self):
        text = "<final_answer>\nline1\nline2\n</final_answer>"
        assert extract_final_answer(text) == "line1\nline2"


class TestExtractBoxed:
    def test_basic(self):
        assert extract_boxed("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\frac{n}{2}}") == "\\frac{n}{2}"

    def test_deeply_nested(self):
        assert extract_boxed("\\boxed{\\frac{1}{\\sqrt{2}}}") == "\\frac{1}{\\sqrt{2}}"

    def test_last_boxed_wins(self):
        text = "First \\boxed{wrong}, then \\boxed{right}"
        assert extract_boxed(text) == "right"

    def test_missing(self):
        with pytest.raises(ValueError):
            extract_boxed("no boxed here")

    def test_unclosed_brace(self):
        with pytest.raises(ValueError):
            extract_boxed("\\boxed{unclosed")

    def test_preserves_inner_whitespace(self):
        assert extract_boxed("\\boxed{  x + 1  }") == "  x + 1  "

    def test_latex_ceiling(self):
        assert (
            extract_boxed("\\boxed{1 + \\left\\lceil \\frac{n}{2} \\right\\rceil}")
            == "1 + \\left\\lceil \\frac{n}{2} \\right\\rceil"
        )

    def test_empty_boxed(self):
        assert extract_boxed("\\boxed{}") == ""

    def test_boxed_in_solution(self):
        text = (
            "Let me work through this...\n"
            "Step 1: compute \\boxed{intermediate}\n"
            "Step 2: therefore \\boxed{\\frac{3}{4}}\n"
        )
        assert extract_boxed(text) == "\\frac{3}{4}"
