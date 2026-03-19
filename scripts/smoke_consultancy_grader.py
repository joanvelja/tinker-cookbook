"""Smoke test: consultancy grader IO path.

Fires a single question+answer through the reference-blind LLM judge
via OpenRouter and prints the full IO at each stage.

Usage: uv run --env-file .env python scripts/smoke_consultancy_grader.py
"""

import asyncio
import inspect

from tinker_cookbook.recipes.consultancy.builders import (
    ConsultancyGpqaBuilder,
    _CONSULTANCY_SYSTEM,
    _CONSULTANCY_USER,
)


CASES = [
    {
        "label": "correct answer",
        "question": "What is the capital of France?",
        "reference": "Paris",
        "extracted": "Paris",
    },
    {
        "label": "wrong answer",
        "question": "What is the capital of France?",
        "reference": "Paris",
        "extracted": "Lyon",
    },
    {
        "label": "ambiguous / partial",
        "question": "What element has atomic number 79?",
        "reference": "Gold (Au)",
        "extracted": "It's a yellow metal used in jewelry",
    },
]


async def main() -> None:
    # Get the default grader config from the builder and build it
    sig = inspect.signature(ConsultancyGpqaBuilder)
    grader_config = sig.parameters["grader_config"].default
    grader = grader_config.build(concurrency_hint=1)

    print("=" * 60)
    print("CONSULTANCY GRADER SMOKE TEST")
    print("=" * 60)
    print(f"Model:        {grader_config.client.model}")
    print(f"Base URL:     {grader_config.client.base_url}")
    print(f"API key env:  {grader_config.client.api_key_env}")
    print(f"Temp:         {grader_config.client.temperature}")
    print(f"Max tokens:   {grader_config.client.max_tokens}")
    print(f"Decision tag: <{grader_config.decision_tag}>")
    print(f"System (raw):         {_CONSULTANCY_SYSTEM!r}")
    print(f"System (after build): {grader.config.system_prompt!r}")
    print(f"Stop (after build):   {grader.config.client.stop!r}")
    print(f"Template:             {_CONSULTANCY_USER!r}")
    print()

    for case in CASES:
        print("-" * 60)
        print(f"Case: {case['label']}")
        print(f"  Question:  {case['question']}")
        print(f"  Reference: {case['reference']}  (NOT sent to judge)")
        print(f"  Extracted: {case['extracted']}")

        result = await grader.grade(
            question=case["question"],
            reference=case["reference"],
            extracted=case["extracted"],
        )
        print(f"  Result: correct={result.correct}, status={result.status}")
        if result.detail:
            print(f"  Detail: {result.detail!r}")
        print()

    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
