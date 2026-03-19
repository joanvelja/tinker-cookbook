#!/usr/bin/env python3
"""Test grader reasoning summary under various API configs.

Run: OPENAI_API_KEY=... uv run python scripts/_test_grader_reasoning.py
"""

import asyncio
import json
from openai import AsyncOpenAI


SYSTEM = (
    "You are a grading assistant. Compare the student's response to the target answer. "
    "Respond with exactly one word: CORRECT or INCORRECT."
)

USER = (
    "Question: What is the number of stars per unit distance proportional to, "
    "given parallax distribution 1/plx^5?\n\n"
    "Target answer: ~ r^3\n\n"
    "Student response: The number of stars per unit distance is proportional to r^3. "
    "This follows from the change of variables: N(r) = N(varpi) |dvarpi/dr| = "
    "C * varpi^{-5} * r^{-2} = C * r^5 * r^{-2} = C * r^3.\n\n"
    "Verdict:"
)

# A harder case where the grader might disagree
USER_WRONG = (
    "Question: What is the number of stars per unit distance proportional to, "
    "given parallax distribution 1/plx^5?\n\n"
    "Target answer: ~ r^3\n\n"
    "Student response: The answer is r^2, since the volume element of a "
    "spherical shell is 4*pi*r^2*dr.\n\n"
    "Verdict:"
)

CONFIGS = [
    # (label, model, effort, summary, input_text)
    ("gpt-5-mini / medium / summary=auto / correct", "gpt-5-mini", "medium", "auto", USER),
    ("gpt-5-mini / medium / summary=auto / wrong", "gpt-5-mini", "medium", "auto", USER_WRONG),
    ("gpt-5-mini / low / summary=auto / correct", "gpt-5-mini", "low", "auto", USER),
    ("gpt-5-mini / medium / no summary / correct", "gpt-5-mini", "medium", None, USER),
    ("gpt-5-mini / minimal / summary=auto / correct", "gpt-5-mini", "minimal", "auto", USER),
    ("gpt-5-mini / medium / summary=concise / wrong", "gpt-5-mini", "medium", "concise", USER_WRONG),
    ("gpt-5-mini / medium / summary=detailed / wrong", "gpt-5-mini", "medium", "detailed", USER_WRONG),
]


async def test_config(client: AsyncOpenAI, label: str, model: str, effort: str, summary: str | None, user: str):
    reasoning_kwargs = {"effort": effort}
    if summary is not None:
        reasoning_kwargs["summary"] = summary

    try:
        response = await client.responses.create(
            model=model,
            instructions=SYSTEM,
            input=user,
            reasoning=reasoning_kwargs,
            max_output_tokens=500,
        )
    except Exception as e:
        return {"label": label, "error": str(e)[:200]}

    result = {
        "label": label,
        "model": model,
        "effort": effort,
        "summary_setting": summary,
        "output_text": response.output_text,
        "output_items": [],
    }

    for item in response.output:
        if item.type == "reasoning":
            summaries = []
            for s in (item.summary or []):
                summaries.append({"type": s.type, "text": s.text})
            result["output_items"].append({
                "type": "reasoning",
                "summary_count": len(summaries),
                "summaries": summaries,
            })
        elif item.type == "message":
            texts = [c.text for c in item.content if hasattr(c, "text")]
            result["output_items"].append({
                "type": "message",
                "texts": texts,
            })

    return result


async def main():
    client = AsyncOpenAI()

    print("Testing grader reasoning under various configs...\n")

    for label, model, effort, summary, user in CONFIGS:
        print(f"--- {label} ---")
        result = await test_config(client, label, model, effort, summary, user)

        if "error" in result:
            print(f"  ERROR: {result['error']}\n")
            continue

        print(f"  output_text: {repr(result['output_text'][:100])}")
        for item in result["output_items"]:
            if item["type"] == "reasoning":
                print(f"  reasoning: {item['summary_count']} summaries")
                for s in item["summaries"]:
                    text = s["text"]
                    print(f"    [{s['type']}] {text[:200]}{'...' if len(text) > 200 else ''}")
            elif item["type"] == "message":
                print(f"  message: {item['texts']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
