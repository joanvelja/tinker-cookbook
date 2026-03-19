"""Analyze reasoning quality in RLVR transcripts: early vs late."""

import re
import json
import sys
from pathlib import Path
from html import unescape
from dataclasses import dataclass, field

RUN_DIR = Path("logs/gpqa-experiment/gpqa-g8-s42")


@dataclass
class Response:
    problem_snippet: str
    response_text: str
    reward_lines: list[str] = field(default_factory=list)
    group_idx: int = 0
    response_idx: int = 0

    @property
    def length(self) -> int:
        return len(self.response_text)

    @property
    def has_final_answer(self) -> bool:
        return "<final_answer>" in self.response_text

    @property
    def has_think_tags(self) -> bool:
        return "<think>" in self.response_text

    @property
    def think_content(self) -> str:
        m = re.search(r"<think>(.*?)</think>", self.response_text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @property
    def final_answer_content(self) -> str:
        m = re.search(r"<final_answer>(.*?)</final_answer>", self.response_text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @property
    def metacognitive_markers(self) -> list[str]:
        markers = []
        patterns = [
            r"[Bb]ut wait", r"[Aa]ctually", r"[Ll]et me reconsider",
            r"[Hh]mm", r"[Ww]ait,", r"[Oo]n second thought",
            r"[Ll]et me think again", r"[Cc]orrection:", r"[Nn]o,? that",
            r"[Ii] was wrong", r"[Tt]hat's not right", r"[Ll]et me re-",
            r"[Aa]lternatively", r"[Hh]old on",
        ]
        for p in patterns:
            found = re.findall(p, self.response_text)
            markers.extend(found)
        return markers

    @property
    def self_correction_count(self) -> int:
        return len(self.metacognitive_markers)

    @property
    def is_structured(self) -> bool:
        """Has step-by-step, numbered lists, bullet points, etc."""
        indicators = [
            r"[Ss]tep \d", r"\d\.\s", r"^\s*-\s", r"^\s*\*\s",
            r"[Ff]irst,", r"[Ss]econd,", r"[Tt]hird,",
            r"[Ff]inally,", r"[Tt]herefore,", r"[Ii]n conclusion",
        ]
        count = sum(1 for p in indicators if re.search(p, self.response_text, re.MULTILINE))
        return count >= 2

    @property
    def reward_correct(self) -> float | None:
        for line in self.reward_lines:
            if "correct" in line.lower():
                m = re.search(r"[\d.]+", line)
                if m:
                    return float(m.group())
        return None

    @property
    def reward_format(self) -> float | None:
        for line in self.reward_lines:
            if "format" in line.lower():
                m = re.search(r"[\d.]+", line)
                if m:
                    return float(m.group())
        return None


def parse_html_trace(filepath: Path) -> list[Response]:
    """Extract responses from an HTML trace file."""
    text = filepath.read_text()
    text = unescape(text)

    # Find all Problem/Response pairs
    responses = []

    # Split by "Problem:" markers in <p class="lt-p"> tags
    # The pattern is: Problem: ...\nResponse: ...\n<reward lines>
    parts = re.split(r'<p class="lt-p">\s*\n?Problem:', text)

    for i, part in enumerate(parts[1:], 1):  # skip first (before any Problem)
        # Extract the problem text (up to </p>)
        problem_match = re.match(r'(.*?)</p>', part, re.DOTALL)
        if not problem_match:
            continue
        problem_text = problem_match.group(1).strip()[:200]

        # Find the Response
        resp_match = re.search(r'Response:\s*(.*?)</p>', part, re.DOTALL)
        if not resp_match:
            continue
        response_text = resp_match.group(1).strip()

        # Find reward lines (typically in <p class="lt-p"> after the response)
        # Look for lines with reward/correct/format
        reward_lines = []
        reward_matches = re.findall(r'<span class="(?:answer|reward)"[^>]*>(.*?)</span>', part)
        for rm in reward_matches:
            reward_lines.append(rm.strip())

        # Also look for plain-text reward lines
        reward_text_matches = re.findall(r'<p class="lt-p">\s*\n?((?:correct|format|reward).*?)</p>', part, re.DOTALL | re.IGNORECASE)
        for rm in reward_text_matches:
            reward_lines.append(rm.strip())

        resp = Response(
            problem_snippet=problem_text,
            response_text=response_text,
            reward_lines=reward_lines,
            group_idx=(i - 1) // 8,  # assuming group_size=8
            response_idx=(i - 1) % 8,
        )
        responses.append(resp)

    return responses


def analyze_responses(responses: list[Response], label: str):
    print(f"\n{'='*80}")
    print(f"  {label}: {len(responses)} responses")
    print(f"{'='*80}")

    # Summary stats
    lengths = [r.length for r in responses]
    structured_count = sum(1 for r in responses if r.is_structured)
    final_answer_count = sum(1 for r in responses if r.has_final_answer)
    think_count = sum(1 for r in responses if r.has_think_tags)
    metacog_counts = [r.self_correction_count for r in responses]

    print(f"\n  Total responses: {len(responses)}")
    print(f"  Avg length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min/Max length: {min(lengths)} / {max(lengths)}")
    print(f"  Structured reasoning: {structured_count}/{len(responses)} ({100*structured_count/len(responses):.0f}%)")
    print(f"  Has <think> tags: {think_count}/{len(responses)} ({100*think_count/len(responses):.0f}%)")
    print(f"  Has <final_answer>: {final_answer_count}/{len(responses)} ({100*final_answer_count/len(responses):.0f}%)")
    print(f"  Avg self-correction markers: {sum(metacog_counts)/len(metacog_counts):.1f}")
    print(f"  Max self-correction markers: {max(metacog_counts)}")

    # Detailed analysis of first 10 responses
    print(f"\n  --- Detailed analysis (first 10) ---")
    for i, r in enumerate(responses[:10]):
        print(f"\n  [{i}] Problem: {r.problem_snippet[:120]}...")
        print(f"      Length: {r.length} chars")
        print(f"      Structured: {r.is_structured}")
        print(f"      <think> tags: {r.has_think_tags}")
        print(f"      <final_answer>: {r.has_final_answer}")
        if r.has_final_answer:
            print(f"      Answer: {r.final_answer_content[:100]}")
        print(f"      Self-corrections: {r.self_correction_count}")
        if r.metacognitive_markers:
            print(f"      Markers: {r.metacognitive_markers[:5]}")
        print(f"      Rewards: {r.reward_lines[:3]}")

        # First 200 chars of response
        print(f"      Response (first 200): {r.response_text[:200]}")

    return responses


def find_format_failures(responses: list[Response]) -> list[Response]:
    """Find responses where format_boxed=0 (no final_answer tags)."""
    return [r for r in responses if not r.has_final_answer]


def detect_patterns(responses: list[Response], label: str):
    """Look for degeneration patterns."""
    print(f"\n  --- Pattern Detection: {label} ---")

    # 1. Guess and box: short responses with minimal reasoning
    short_threshold = 500
    short = [r for r in responses if r.length < short_threshold]
    print(f"  'Guess and box' (< {short_threshold} chars): {len(short)}/{len(responses)}")

    # 2. Template lock: check similarity of response structures
    # Use first line of think content as proxy
    think_starts = [r.think_content[:100] for r in responses if r.think_content]
    if think_starts:
        # Check for common prefixes
        from collections import Counter
        prefix_counter = Counter()
        for ts in think_starts:
            # Normalize: first 50 chars
            prefix_counter[ts[:50]] += 1
        most_common = prefix_counter.most_common(3)
        print(f"  Template lock check - most common think prefixes:")
        for prefix, count in most_common:
            print(f"    ({count}x) \"{prefix[:60]}...\"")

    # 3. Post-answer waste: content after final_answer
    waste_count = 0
    for r in responses:
        if r.has_final_answer:
            after_answer = r.response_text.split("</final_answer>")[-1].strip()
            if len(after_answer) > 50:
                waste_count += 1
    print(f"  Post-answer waste (>50 chars after </final_answer>): {waste_count}/{len(responses)}")

    # 4. Dead-end spiraling: responses with many self-corrections but wrong answer
    spiral_count = sum(1 for r in responses if r.self_correction_count >= 3)
    print(f"  Potential dead-end spiraling (>=3 corrections): {spiral_count}/{len(responses)}")


def analyze_format_failures(responses: list[Response], label: str):
    """Analyze WHY format failures happen."""
    failures = find_format_failures(responses)
    print(f"\n  --- Format Failures: {label} ({len(failures)} total) ---")

    for i, r in enumerate(failures[:3]):
        print(f"\n  [Failure {i}]")
        print(f"    Problem: {r.problem_snippet[:120]}...")
        print(f"    Length: {r.length} chars")

        # Check if truncated
        if r.response_text.endswith("...") or len(r.response_text) > 4000:
            print(f"    LIKELY TRUNCATED")

        # Check for alternative formats
        has_boxed = "\\boxed" in r.response_text
        has_answer_is = re.search(r"[Tt]he answer is", r.response_text) is not None
        has_therefore = re.search(r"[Tt]herefore.*?[A-D]", r.response_text) is not None
        has_plain_answer = re.search(r"(?:^|\n)\s*[A-D]\s*$", r.response_text, re.MULTILINE) is not None

        print(f"    Has \\boxed: {has_boxed}")
        print(f"    Has 'the answer is': {has_answer_is}")
        print(f"    Has 'therefore [A-D]': {has_therefore}")
        print(f"    Has standalone letter answer: {has_plain_answer}")

        # Last 300 chars
        print(f"    Last 300 chars: ...{r.response_text[-300:]}")
        print(f"    Rewards: {r.reward_lines[:5]}")


def main():
    early_file = RUN_DIR / "train_iteration_000000.html"
    late_file = RUN_DIR / "train_iteration_000018.html"

    print("Parsing early trace (iteration 0)...")
    early = parse_html_trace(early_file)
    print(f"  Found {len(early)} responses")

    print("Parsing late trace (iteration 18)...")
    late = parse_html_trace(late_file)
    print(f"  Found {len(late)} responses")

    early_analyzed = analyze_responses(early, "EARLY (Iteration 0)")
    late_analyzed = analyze_responses(late, "LATE (Iteration 18)")

    detect_patterns(early, "EARLY")
    detect_patterns(late, "LATE")

    analyze_format_failures(early, "EARLY")
    analyze_format_failures(late, "LATE")

    # Comparative summary
    print(f"\n{'='*80}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*80}")

    e_len = sum(r.length for r in early) / len(early) if early else 0
    l_len = sum(r.length for r in late) / len(late) if late else 0
    print(f"  Avg response length: {e_len:.0f} (early) -> {l_len:.0f} (late) [{l_len-e_len:+.0f}]")

    e_struct = sum(1 for r in early if r.is_structured) / len(early) * 100 if early else 0
    l_struct = sum(1 for r in late if r.is_structured) / len(late) * 100 if late else 0
    print(f"  Structured: {e_struct:.0f}% (early) -> {l_struct:.0f}% (late)")

    e_think = sum(1 for r in early if r.has_think_tags) / len(early) * 100 if early else 0
    l_think = sum(1 for r in late if r.has_think_tags) / len(late) * 100 if late else 0
    print(f"  <think> tags: {e_think:.0f}% (early) -> {l_think:.0f}% (late)")

    e_fa = sum(1 for r in early if r.has_final_answer) / len(early) * 100 if early else 0
    l_fa = sum(1 for r in late if r.has_final_answer) / len(late) * 100 if late else 0
    print(f"  <final_answer>: {e_fa:.0f}% (early) -> {l_fa:.0f}% (late)")

    e_mc = sum(r.self_correction_count for r in early) / len(early) if early else 0
    l_mc = sum(r.self_correction_count for r in late) / len(late) if late else 0
    print(f"  Avg self-corrections: {e_mc:.1f} (early) -> {l_mc:.1f} (late)")


if __name__ == "__main__":
    main()
