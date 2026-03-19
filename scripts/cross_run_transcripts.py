"""Read transcripts from both rungs for questions where thinking helps most.

For each target: grab one WINNING episode from rung2 and one LOSING episode from rung1,
then extract the think blocks and public arguments.
"""

import json
import re
from pathlib import Path

BASE = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/thinking-experiment")
RUNG1 = BASE / "rung1-no-think" / "episodes" / "episodes.jsonl"
RUNG2 = BASE / "rung2-private-think" / "episodes" / "episodes.jsonl"


def load_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_think_block(text: str) -> str | None:
    """Extract content between <thinking> tags."""
    m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_public(text: str) -> str:
    """Extract text outside <thinking> tags."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def print_transcript(ep: dict, max_think: int = 600, max_public: int = 600):
    """Print transcript turns from an episode."""
    for j, turn in enumerate(ep["transcript"]):
        role = turn.get("role", "?")
        phase = turn.get("phase", "?")
        rnd = turn.get("round", "?")
        text = turn.get("text", "")

        if not text:
            continue

        think = extract_think_block(text)
        public = extract_public(text)
        print(f"\n  Turn {j} [{role}, phase={phase}, round={rnd}]:")
        if think:
            tlen = len(think)
            print(f"    <thinking> ({tlen} chars):")
            # Show first N chars
            snippet = think[:max_think]
            for line in snippet.split("\n"):
                print(f"      {line}")
            if tlen > max_think:
                print(f"      [...{tlen - max_think} more chars...]")
        if public:
            plen = len(public)
            snippet = public[:max_public]
            for line in snippet.split("\n"):
                print(f"    {line}")
            if plen > max_public:
                print(f"    [...{plen - max_public} more chars...]")


def main():
    with open(BASE / "thinking_helps_targets.json") as f:
        targets = json.load(f)

    ep1 = load_episodes(RUNG1)
    ep2 = load_episodes(RUNG2)

    # Index by target
    by_target_1: dict[str, list[dict]] = {}
    by_target_2: dict[str, list[dict]] = {}
    for ep in ep1:
        by_target_1.setdefault(ep["target"], []).append(ep)
    for ep in ep2:
        by_target_2.setdefault(ep["target"], []).append(ep)

    for i, target in enumerate(targets[:5]):
        print(f"\n{'#'*80}")
        print(f"# TARGET {i+1}")
        print(f"{'#'*80}")
        tdisp = target[:200] + "..." if len(target) > 200 else target
        print(f"Target: {tdisp}")

        # ── Rung1: find a LOSING episode with non-empty transcript ──
        eps1 = by_target_1.get(target, [])
        losing1 = [e for e in eps1 if e["reward"] <= 0]
        winning1 = [e for e in eps1 if e["reward"] > 0]
        print(f"\nRung1 (no-think): {len(eps1)} episodes, {len(winning1)} wins, {len(losing1)} losses")

        # Find one with actual transcript content
        shown = False
        for ep in losing1:
            has_content = any(t.get("text", "") for t in ep["transcript"])
            if has_content:
                print(f"\n--- Rung1 LOSING episode (role={ep['role']}, reward={ep['reward']}) ---")
                print(f"Winner: {ep['winner']}")
                print(f"Answers: a={ep['answers'].get('public_debater_a', 'N/A')[:120]}")
                print(f"         b={ep['answers'].get('public_debater_b', 'N/A')[:120]}")
                print_transcript(ep)
                shown = True
                break
        if not shown:
            print("  (No losing episode with non-empty transcript found)")

        # ── Rung2: find a WINNING episode with non-empty transcript ──
        eps2 = by_target_2.get(target, [])
        winning2 = [e for e in eps2 if e["reward"] > 0]
        losing2 = [e for e in eps2 if e["reward"] <= 0]
        print(f"\nRung2 (private-think): {len(eps2)} episodes, {len(winning2)} wins, {len(losing2)} losses")

        shown = False
        for ep in winning2:
            has_content = any(t.get("text", "") for t in ep["transcript"])
            if has_content:
                print(f"\n--- Rung2 WINNING episode (role={ep['role']}, reward={ep['reward']}) ---")
                print(f"Winner: {ep['winner']}")
                print(f"Answers: a={ep['answers'].get('public_debater_a', 'N/A')[:120]}")
                print(f"         b={ep['answers'].get('public_debater_b', 'N/A')[:120]}")
                print(f"Think:   a={str(ep['answers'].get('think_debater_a', 'N/A'))[:80]}")
                print(f"         b={str(ep['answers'].get('think_debater_b', 'N/A'))[:80]}")
                print_transcript(ep)
                shown = True
                break
        if not shown:
            # Try losing if no winning
            for ep in losing2:
                has_content = any(t.get("text", "") for t in ep["transcript"])
                if has_content:
                    print(f"\n--- Rung2 LOSING episode (no wins with transcripts) (role={ep['role']}, reward={ep['reward']}) ---")
                    print_transcript(ep)
                    shown = True
                    break
        if not shown:
            print("  (No rung2 episode with non-empty transcript found)")


if __name__ == "__main__":
    main()
