"""
Judge prompt phrasing experiment.

Tests 5 prompt variants on 3 debates to measure:
- Does the judge produce reasoning before the <decision> tag?
- Chars of reasoning
- Does it cite specific debater arguments?
- Is the verdict correct?
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers.base import Message

DEBATE_IDS = [
    "e2b16db5574e40b0a5786037abe3e9a2",
    "148832ff4b1b4fe2bf121accba4d327c",
    "5de2ca2b72de4fac91830067b478bc0c",
]

JUDGE_SYSTEM = (
    "You are a judge evaluating two responses to an open-ended question. "
    "Do not try to solve the problem yourself. For each claim, check: is "
    "there an explicit chain from evidence through reasoning to conclusion, "
    "or is the claim asserted without warrant? Confident tone and jargon "
    "are not evidence. A specific inference beats a vague appeal to "
    "expertise. Check internal consistency: do the premises actually entail "
    "the conclusion? After the exchange, ask: who addressed the strongest "
    "objection? Who conceded genuine errors and updated? Concession shows "
    "epistemic strength. Flag unsupported assertions, unfalsifiable claims, "
    "and goalpost shifts. Prefer the response whose final answer is most "
    "consistent with its own stated reasoning. Tie only when quality is "
    "genuinely symmetric."
)

PROMPT_VARIANTS = {
    "baseline": "Briefly analyze the key arguments, then give your decision.",
    "walk_through": (
        "Walk through the strongest argument from each side in your public response, "
        "then give your decision."
    ),
    "step_by_step": (
        "Reason step by step about whose scientific reasoning is more sound, "
        "then give your decision."
    ),
    "compare_claims": (
        "In your public analysis, compare the key claims from each debater. "
        "Which debater's reasoning chain is more internally consistent? "
        "Give your decision at the end."
    ),
    "structured_eval": (
        "Evaluate the debate publicly. For each debater: (1) state their core claim, "
        "(2) assess whether they supported it with evidence, (3) note if they addressed "
        "the opponent's strongest point. Then give your decision."
    ),
}

FIELD_INSTRUCTION = '\n\nYou MUST end your response with a <decision> tag containing exactly one of: A, B, or tie.'


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_debates() -> dict[str, dict]:
    """Load the 3 test debates from episodes.jsonl."""
    episodes_path = Path("logs/protocol-experiment/seq-v1/episodes/episodes.jsonl")
    debates = {}
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            if ep["debate_id"] in DEBATE_IDS and ep.get("role") == "debater_a":
                debates[ep["debate_id"]] = ep
    return debates


def build_judge_messages(debate: dict, judge_user_prompt: str) -> list[Message]:
    """Build the judge's message list from a debate episode."""
    messages: list[Message] = [
        Message(role="system", content=JUDGE_SYSTEM),
    ]

    # Build transcript as user messages with wrapped debater responses
    transcript = debate["transcript"]
    transcript_text_parts = []
    for turn in transcript:
        raw = turn["text"]
        public = strip_thinking(raw)
        label = "A" if turn["role"] == "debater_a" else "B"
        wrapped = f'<response from="Expert {label}">\n{public}\n</response>'
        transcript_text_parts.append(wrapped)

    # Consolidate all transcript into a single user message
    transcript_msg = "\n\n".join(transcript_text_parts)
    messages.append(Message(role="user", content=transcript_msg))

    # Judge instruction (the variant)
    messages.append(Message(role="user", content=judge_user_prompt + FIELD_INSTRUCTION))

    return messages


def analyze_response(text: str, debate: dict) -> dict[str, Any]:
    """Analyze a judge response for quality metrics."""
    # Find decision tag
    decision_match = re.search(r"<decision>\s*(.*?)\s*</decision>", text, re.IGNORECASE)
    has_decision = decision_match is not None
    decision = decision_match.group(1).strip() if decision_match else None

    # Reasoning before decision
    if decision_match:
        reasoning_before = text[:decision_match.start()].strip()
    else:
        reasoning_before = text.strip()

    has_reasoning = len(reasoning_before) > 20
    reasoning_chars = len(reasoning_before)

    # Does it cite specific debater arguments?
    cites_a = bool(re.search(r"Expert A|Debater A|Expert A's", text, re.IGNORECASE))
    cites_b = bool(re.search(r"Expert B|Debater B|Expert B's", text, re.IGNORECASE))
    cites_specific = cites_a and cites_b

    # Check verdict correctness
    ground_truth_winner = debate.get("winner")
    if decision:
        decision_lower = decision.lower().strip()
        if decision_lower == "a":
            predicted_winner = "debater_a"
        elif decision_lower == "b":
            predicted_winner = "debater_b"
        elif decision_lower == "tie":
            predicted_winner = None
        else:
            predicted_winner = "PARSE_FAIL"
    else:
        predicted_winner = "NO_DECISION"

    correct = predicted_winner == ground_truth_winner

    return {
        "has_reasoning": has_reasoning,
        "reasoning_chars": reasoning_chars,
        "cites_specific": cites_specific,
        "cites_a": cites_a,
        "cites_b": cites_b,
        "has_decision": has_decision,
        "decision": decision,
        "predicted_winner": predicted_winner,
        "ground_truth_winner": ground_truth_winner,
        "correct": correct,
    }


def run_experiment():
    tok = get_tokenizer("Qwen/Qwen3.5-27B")
    renderer = get_renderer("qwen3_5_disable_thinking", tok)
    service = tinker.ServiceClient()
    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.5-27B")

    debates = load_debates()
    print(f"Loaded {len(debates)} debates")

    results: list[dict] = []

    total = len(PROMPT_VARIANTS) * len(debates)
    idx = 0

    for variant_name, variant_prompt in PROMPT_VARIANTS.items():
        for debate_id, debate in debates.items():
            idx += 1
            print(f"  [{idx}/{total}] variant={variant_name}, debate={debate_id[:8]}")

            messages = build_judge_messages(debate, variant_prompt)
            model_input = renderer.build_generation_prompt(messages)
            stop_seqs = renderer.get_stop_sequences()

            t0 = time.monotonic()
            response = sampler.sample(
                model_input,
                max_new_tokens=2048,
                temperature=0.0,
                stop_sequences=stop_seqs,
            )
            elapsed = time.monotonic() - t0

            # Parse response
            raw_tokens = list(response.tokens)
            parsed_msg, parse_ok = renderer.parse_response(raw_tokens)

            if isinstance(parsed_msg["content"], str):
                response_text = parsed_msg["content"]
            else:
                response_text = "".join(
                    p.get("text", "") for p in parsed_msg["content"] if p.get("type") == "text"
                )

            analysis = analyze_response(response_text, debate)

            result = {
                "variant": variant_name,
                "debate_id": debate_id[:8],
                **analysis,
                "elapsed_s": round(elapsed, 1),
                "full_response": response_text,
            }
            results.append(result)

            # Print inline result
            status = "OK" if analysis["correct"] else "WRONG"
            reasoning = "REASONING" if analysis["has_reasoning"] else "BARE"
            print(f"    -> {reasoning} ({analysis['reasoning_chars']} chars), "
                  f"decision={analysis['decision']}, {status}, "
                  f"cites_both={analysis['cites_specific']}, {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Variant':<20} {'Reasoning%':>10} {'Avg Chars':>10} {'Cites Both%':>12} {'Correct%':>10} {'Parse%':>8}")
    print("-" * 120)

    for variant_name in PROMPT_VARIANTS:
        vr = [r for r in results if r["variant"] == variant_name]
        n = len(vr)
        reasoning_pct = sum(1 for r in vr if r["has_reasoning"]) / n * 100
        avg_chars = sum(r["reasoning_chars"] for r in vr) / n
        cites_pct = sum(1 for r in vr if r["cites_specific"]) / n * 100
        correct_pct = sum(1 for r in vr if r["correct"]) / n * 100
        parse_pct = sum(1 for r in vr if r["has_decision"]) / n * 100
        print(f"{variant_name:<20} {reasoning_pct:>9.0f}% {avg_chars:>10.0f} {cites_pct:>11.0f}% {correct_pct:>9.0f}% {parse_pct:>7.0f}%")

    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)
    for r in results:
        print(f"\n--- {r['variant']} x {r['debate_id']} ---")
        print(f"  Reasoning: {r['has_reasoning']} ({r['reasoning_chars']} chars)")
        print(f"  Decision: {r['decision']} (GT: {r['ground_truth_winner']}) -> {'CORRECT' if r['correct'] else 'WRONG'}")
        print(f"  Cites both: {r['cites_specific']}")
        print(f"  Preview: {r['full_response'][:400]}")

    # Save full results
    out_path = Path("logs/judge_prompt_experiment_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
