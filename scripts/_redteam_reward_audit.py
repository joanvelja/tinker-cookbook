"""Red-team audit of reward signal integrity for RLVR GPQA runs.

Parses HTML trace files, extracts episodes, and checks for grader errors.
"""

import re
import sys
import html
from pathlib import Path
from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass
class Episode:
    """One rollout within a trace file."""
    trace_file: str
    episode_idx: int
    question_snippet: str  # first 120 chars
    response_snippet: str  # first 200 chars
    extracted_answer: str | None  # what was inside <final_answer> tags
    reference: str
    boxed: bool
    eos: bool
    correct: bool
    reward: float
    grade_status: str  # from the table


_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_REWARD_RE = re.compile(
    r"Boxed:\s*(✓|✗),\s*EOS:\s*(✓|✗),\s*Correct:\s*(✓|✗),\s*Reward:\s*([-\d.]+)"
)


def parse_trace(filepath: Path) -> list[Episode]:
    """Parse an HTML trace file into Episode objects."""
    text = filepath.read_text()
    soup = BeautifulSoup(text, "html.parser")

    episodes = []
    paragraphs = soup.find_all("p", class_="lt-p")

    # Episodes come in groups of 3 paragraphs: Problem, Response, Reference + Reward line
    # Actually looking at the structure: Problem paragraph, Response paragraph, Reference paragraph, Reward paragraph
    # Let's find them by content pattern matching

    questions = []
    responses = []
    references = []
    rewards_data = []

    i = 0
    while i < len(paragraphs):
        p_text = paragraphs[i].get_text(strip=True)

        if p_text.startswith("Problem:"):
            questions.append(p_text[len("Problem:"):].strip())
            # Next paragraph is Response
            if i + 1 < len(paragraphs):
                resp_text = paragraphs[i + 1].get_text(strip=True)
                if resp_text.startswith("Response:"):
                    responses.append(resp_text[len("Response:"):].strip())
                else:
                    responses.append("")
            else:
                responses.append("")
            # Next is Reference Answer
            if i + 2 < len(paragraphs):
                ref_text = paragraphs[i + 2].get_text(strip=True)
                if ref_text.startswith("Reference Answer:"):
                    references.append(ref_text[len("Reference Answer:"):].strip())
                else:
                    references.append("???")
            else:
                references.append("???")
            # Next is reward line
            if i + 3 < len(paragraphs):
                rw_text = paragraphs[i + 3].get_text(strip=True)
                m = _REWARD_RE.search(rw_text)
                if m:
                    rewards_data.append({
                        "boxed": m.group(1) == "✓",
                        "eos": m.group(2) == "✓",
                        "correct": m.group(3) == "✓",
                        "reward": float(m.group(4)),
                    })
                    i += 4
                    continue
                else:
                    rewards_data.append({"boxed": False, "eos": False, "correct": False, "reward": -999.0})
                    i += 3
                    continue
            i += 3
            continue
        i += 1

    # Parse grade_status from trajectory tables
    grade_statuses = []
    tables = soup.find_all("table", class_="lt-table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 5:
                step_val = cells[0].get_text(strip=True)
                if step_val == "0":  # The actual step row (not "final" or "total")
                    grade_statuses.append(cells[4].get_text(strip=True))

    # Build episodes
    n = min(len(questions), len(responses), len(references), len(rewards_data))
    for idx in range(n):
        resp = responses[idx]
        # Try to extract the <final_answer> from the response
        # The HTML may have escaped the tags
        resp_for_extract = resp
        m = _FINAL_ANSWER_RE.search(resp_for_extract)
        extracted = m.group(1).strip() if m else None

        gs = grade_statuses[idx] if idx < len(grade_statuses) else "unknown"

        episodes.append(Episode(
            trace_file=filepath.name,
            episode_idx=idx,
            question_snippet=questions[idx][:120],
            response_snippet=resp[:200],
            extracted_answer=extracted,
            reference=references[idx],
            boxed=rewards_data[idx]["boxed"],
            eos=rewards_data[idx]["eos"],
            correct=rewards_data[idx]["correct"],
            reward=rewards_data[idx]["reward"],
            grade_status=gs,
        ))

    return episodes


def audit_episode(ep: Episode) -> dict:
    """Evaluate a single episode for grader correctness.

    Returns a dict with audit findings.
    """
    findings = {
        "trace": ep.trace_file,
        "idx": ep.episode_idx,
        "ref": ep.reference,
        "extracted": ep.extracted_answer,
        "grader_said_correct": ep.correct,
        "reward": ep.reward,
        "grade_status": ep.grade_status,
        "format_ok": ep.boxed,
        "eos_ok": ep.eos,
    }

    # Check format failures
    if not ep.boxed:
        if ep.extracted_answer is not None:
            findings["issue"] = "ANOMALY: boxed=False but extraction succeeded"
        else:
            findings["issue"] = "FORMAT_FAIL: no <final_answer> tags"

    if ep.grade_status == "ambiguous":
        findings["issue"] = "AMBIGUOUS_GRADE"

    if ep.grade_status == "error":
        findings["issue"] = "GRADE_ERROR"

    return findings


def main():
    run_dir = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/gpqa-experiment/gpqa-g8-s42")

    # Pick trace files: early (iter 0), middle (iter 9), and late (iter 18)
    trace_files = [
        "train_iteration_000000.html",
        "train_iteration_000009.html",
        "train_iteration_000018.html",
        "eval_test_iteration_000000.html",
    ]

    all_episodes = []
    for tf in trace_files:
        fp = run_dir / tf
        if not fp.exists():
            print(f"SKIP: {tf} not found")
            continue
        episodes = parse_trace(fp)
        all_episodes.extend(episodes)
        print(f"\n{'='*80}")
        print(f"TRACE: {tf} — {len(episodes)} episodes parsed")
        print(f"{'='*80}")

    # Stats
    total = len(all_episodes)
    if total == 0:
        print("No episodes found!")
        return

    correct_count = sum(1 for e in all_episodes if e.correct)
    format_fail = sum(1 for e in all_episodes if not e.boxed)
    eos_fail = sum(1 for e in all_episodes if not e.eos)
    ambiguous = sum(1 for e in all_episodes if e.grade_status == "ambiguous")
    errors = sum(1 for e in all_episodes if e.grade_status == "error")

    print(f"\n{'='*80}")
    print("AGGREGATE STATS")
    print(f"{'='*80}")
    print(f"Total episodes: {total}")
    print(f"Correct: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"Format failures (no <final_answer>): {format_fail}/{total} ({100*format_fail/total:.1f}%)")
    print(f"EOS failures (truncated): {eos_fail}/{total} ({100*eos_fail/total:.1f}%)")
    print(f"Ambiguous grades: {ambiguous}/{total}")
    print(f"Error grades: {errors}/{total}")

    # Grade status breakdown
    status_counts: dict[str, int] = {}
    for e in all_episodes:
        status_counts[e.grade_status] = status_counts.get(e.grade_status, 0) + 1
    print(f"\nGrade status distribution: {status_counts}")

    # Show detailed episodes — sample for human review
    print(f"\n{'='*80}")
    print("DETAILED EPISODE SAMPLE (for manual grading verification)")
    print(f"{'='*80}")

    # Show all unique question/reference/extracted combos
    seen = set()
    sample_count = 0
    for ep in all_episodes:
        key = (ep.question_snippet[:60], ep.reference, ep.extracted_answer, ep.correct)
        if key in seen:
            continue
        seen.add(key)
        sample_count += 1

        print(f"\n--- Episode {ep.trace_file}:{ep.episode_idx} ---")
        print(f"  Question: {ep.question_snippet}...")
        print(f"  Reference: {ep.reference}")
        print(f"  Extracted: {ep.extracted_answer}")
        print(f"  Grader verdict: {'CORRECT' if ep.correct else 'INCORRECT'}")
        print(f"  Grade status: {ep.grade_status}")
        print(f"  Format OK: {ep.boxed}, EOS OK: {ep.eos}")
        print(f"  Reward: {ep.reward}")

        # My assessment
        if ep.extracted_answer is not None and ep.reference:
            # Normalize for comparison
            ext_norm = ep.extracted_answer.strip().lower()
            ref_norm = ep.reference.strip().lower()
            if ext_norm == ref_norm:
                match = "EXACT_MATCH"
            elif ref_norm in ext_norm or ext_norm in ref_norm:
                match = "SUBSTRING_MATCH"
            else:
                match = "NO_MATCH"
            print(f"  String comparison: {match}")

            # Flag potential grader errors
            if match in ("EXACT_MATCH", "SUBSTRING_MATCH") and not ep.correct:
                print(f"  *** POTENTIAL FALSE NEGATIVE: extracted '{ep.extracted_answer}' matches reference '{ep.reference}' but grader said INCORRECT ***")
            elif match == "NO_MATCH" and ep.correct:
                print(f"  *** CHECK: extracted '{ep.extracted_answer}' != reference '{ep.reference}' but grader said CORRECT — may be semantic equivalence or FP ***")

    # Specific checks for the _parse_verdict bug
    print(f"\n{'='*80}")
    print("CHECK: _parse_verdict ambiguity bug")
    print(f"{'='*80}")
    if ambiguous > 0:
        print(f"FOUND {ambiguous} AMBIGUOUS grades! This may indicate the _parse_verdict bug is active.")
        for ep in all_episodes:
            if ep.grade_status == "ambiguous":
                print(f"  {ep.trace_file}:{ep.episode_idx} — ref={ep.reference}, extracted={ep.extracted_answer}")
    else:
        print("No ambiguous grades found. _parse_verdict fix appears to be working.")

    # Check for format failures that might be truncation
    print(f"\n{'='*80}")
    print("FORMAT FAILURE ANALYSIS")
    print(f"{'='*80}")
    format_and_eos_fail = sum(1 for e in all_episodes if not e.boxed and not e.eos)
    format_only_fail = sum(1 for e in all_episodes if not e.boxed and e.eos)
    print(f"Both format + EOS fail (likely truncation): {format_and_eos_fail}")
    print(f"Format fail but EOS ok (model didn't use tags): {format_only_fail}")

    # Reward distribution
    print(f"\n{'='*80}")
    print("REWARD DISTRIBUTION")
    print(f"{'='*80}")
    reward_counts: dict[float, int] = {}
    for ep in all_episodes:
        r = round(ep.reward, 2)
        reward_counts[r] = reward_counts.get(r, 0) + 1
    for r in sorted(reward_counts.keys()):
        print(f"  reward={r}: {reward_counts[r]} episodes")

    # Per-trace breakdown
    print(f"\n{'='*80}")
    print("PER-TRACE BREAKDOWN")
    print(f"{'='*80}")
    trace_names = sorted(set(ep.trace_file for ep in all_episodes))
    for tn in trace_names:
        trace_eps = [e for e in all_episodes if e.trace_file == tn]
        tc = sum(1 for e in trace_eps if e.correct)
        tf = sum(1 for e in trace_eps if not e.boxed)
        print(f"  {tn}: {len(trace_eps)} eps, {tc} correct ({100*tc/len(trace_eps):.1f}%), {tf} format fails")


if __name__ == "__main__":
    main()
