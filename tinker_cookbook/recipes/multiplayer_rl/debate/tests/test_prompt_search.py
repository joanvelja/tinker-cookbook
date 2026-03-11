from __future__ import annotations

import asyncio

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.eval_semantic_prompt_variants import (
    BankCase,
    PromptBlock,
    PromptVariant,
    _raw_case_to_bank_case,
    _evaluate_case,
)


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        (True, True),
        (False, False),
        ("positive", True),
        ("negative", False),
        ("SAME", True),
        ("DIFFERENT", False),
    ],
)
def test_raw_case_to_bank_case_normalizes_string_labels(
    raw_label: bool | str,
    expected: bool,
) -> None:
    case = _raw_case_to_bank_case(
        {
            "id": "case-1",
            "task": "matcher",
            "distribution": "holdout_test",
            "question": "Are these equivalent?",
            "target": None,
            "response": None,
            "a": "Answer A",
            "b": "Answer B",
            "label": raw_label,
            "notes": "repro for string label parsing",
        },
        source="test",
    )

    assert case.label is expected


def test_raw_case_to_bank_case_rejects_unknown_string_labels() -> None:
    with pytest.raises(ValueError, match="Unsupported label"):
        _raw_case_to_bank_case(
            {
                "id": "case-1",
                "task": "matcher",
                "distribution": "holdout_test",
                "question": "Are these equivalent?",
                "target": None,
                "response": None,
                "a": "Answer A",
                "b": "Answer B",
                "label": "maybe",
                "notes": "repro for invalid label parsing",
            },
            source="test",
        )


def test_raw_case_to_bank_case_allows_matcher_target_response_fallback() -> None:
    case = _raw_case_to_bank_case(
        {
            "id": "case-legacy-matcher",
            "task": "matcher",
            "distribution": "legacy_shape",
            "question": "Are these equivalent?",
            "target": "Answer A",
            "response": "Answer B",
            "a": None,
            "b": None,
            "label": True,
            "notes": "legacy matcher shape uses target/response",
        },
        source="test",
    )

    assert case.a == "Answer A"
    assert case.b == "Answer B"


def test_raw_case_to_bank_case_allows_grader_a_b_fallback() -> None:
    case = _raw_case_to_bank_case(
        {
            "id": "case-legacy-grader",
            "task": "grader",
            "distribution": "legacy_shape",
            "question": "Is this correct?",
            "target": None,
            "response": None,
            "a": "Gold answer",
            "b": "Candidate answer",
            "label": False,
            "notes": "legacy grader shape uses a/b",
        },
        source="test",
    )

    assert case.target == "Gold answer"
    assert case.response == "Candidate answer"


class _ExplodingScorer:
    async def complete_binary(self, *, system: str, user: str):
        raise RuntimeError("synthetic scorer failure")


def test_evaluate_case_marks_scorer_exceptions_as_invalid_verdict() -> None:
    variant = PromptVariant(
        name="synthetic",
        matcher=PromptBlock(
            system="sys",
            user='Answer A: "{{ a }}"\nAnswer B: "{{ b }}"',
            positive="SAME",
            negative="DIFFERENT",
        ),
        grader=PromptBlock(
            system="sys",
            user='Correct answer: "{{ target }}"\nResponse: "{{ response }}"',
            positive="CORRECT",
            negative="INCORRECT",
        ),
    )
    case = BankCase(
        id="matcher-error",
        task="matcher",
        distribution="error_path",
        question="Are these equivalent?",
        target=None,
        response=None,
        a="A",
        b="B",
        label=True,
        notes="scorer exception should not abort the whole bench",
    )

    result = asyncio.run(_evaluate_case(asyncio.Semaphore(1), _ExplodingScorer(), variant, case))

    assert result.valid_verdict is False
    assert result.prediction is False
    assert result.verdict is None
    assert "synthetic scorer failure" in result.raw_response
