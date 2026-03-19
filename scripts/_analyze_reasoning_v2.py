"""Analyze reasoning quality in RLVR transcripts: early vs late. V2 - better parsing."""

import re
import json
from pathlib import Path
from html import unescape
from html.parser import HTMLParser
from dataclasses import dataclass, field
from collections import Counter

RUN_DIR = Path("logs/gpqa-experiment/gpqa-g8-s42")


@dataclass
class TrajectoryReward:
    step: str
    ob_len: int | str
    ac_len: int | str
    reward: float
    grade_status: str


@dataclass
class ResponseGroup:
    """A group of responses (rollouts) for one problem."""
    problem_text: str
    responses: list[str]
    trajectory_rewards: list[list[TrajectoryReward]]  # one per response


def parse_reward_tables(text: str) -> list[list[TrajectoryReward]]:
    """Extract trajectory reward tables."""
    results = []
    # Find table captions like "Trajectory 0", "Trajectory 1", etc.
    table_pattern = r'<div class="lt-table-caption">\s*Trajectory (\d+)\s*</div>'
    tables = list(re.finditer(table_pattern, text))

    # For each trajectory, find the preceding table
    # Actually, let's find all tables with reward columns
    table_blocks = re.split(r'<div class="lt-table-caption">', text)

    for block in table_blocks:
        caption_match = re.match(r'\s*Trajectory (\d+)\s*</div>', block)
        if not caption_match:
            continue

        # Find the table BEFORE this caption (it's in the previous block)
        # Actually the table is before the caption div

    # Different approach: extract all tables
    all_tables = re.findall(
        r'<table class="lt-table">(.*?)</table>',
        text, re.DOTALL
    )

    trajectories = []
    for table_html in all_tables:
        # Check if this is a reward table (has 'reward' header)
        if 'reward' not in table_html:
            continue

        rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
        trajectory = []
        for row in rows[1:]:  # skip header
            cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
            cells = [c.strip() for c in cells]
            if len(cells) >= 5:
                try:
                    reward_val = float(cells[3]) if cells[3] != '-' else 0.0
                except:
                    reward_val = 0.0
                trajectory.append(TrajectoryReward(
                    step=cells[0],
                    ob_len=cells[1],
                    ac_len=cells[2],
                    reward=reward_val,
                    grade_status=cells[4],
                ))
        if trajectory:
            trajectories.append(trajectory)

    return trajectories


def parse_html_trace(filepath: Path) -> list[ResponseGroup]:
    """Parse trace into problem groups with responses and rewards."""
    text = filepath.read_text()
    text = unescape(text)

    # Find all section bodies (each contains a problem group)
    # The structure is: do_batched_group_rollout section > problem + 8 responses + reward table
    sections = re.split(r'do_batched_group_rollout', text)

    groups = []
    for section in sections[1:]:  # skip preamble
        # Extract problem text
        problem_match = re.search(r'Problem:\s*(.*?)(?=Response:|$)', section, re.DOTALL)
        if not problem_match:
            continue
        problem_text = problem_match.group(1).strip()
        # Clean up HTML
        problem_text = re.sub(r'<[^>]+>', '', problem_text).strip()

        # Extract responses
        responses = []
        resp_parts = re.split(r'Response:\s*', section)
        for rp in resp_parts[1:]:
            # Response ends at next </p> or next Problem:
            resp_match = re.match(r'(.*?)(?:</p>)', rp, re.DOTALL)
            if resp_match:
                resp_text = resp_match.group(1).strip()
                resp_text = re.sub(r'<[^>]+>', '', resp_text).strip()
                responses.append(resp_text)

        # Extract reward tables
        trajectories = parse_reward_tables(section)

        if responses:
            groups.append(ResponseGroup(
                problem_text=problem_text,
                responses=responses,
                trajectory_rewards=trajectories,
            ))

    return groups


def analyze_response(text: str) -> dict:
    """Analyze a single response."""
    result = {}
    result['length'] = len(text)
    result['has_think'] = '<think>' in text
    result['has_final_answer'] = '<final_answer>' in text

    # Think content
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    result['think_content'] = think_match.group(1).strip() if think_match else ""
    result['think_length'] = len(result['think_content'])

    # Content after </think>
    after_think = text.split('</think>')[-1].strip() if '</think>' in text else text
    result['post_think_length'] = len(after_think)

    # Final answer
    fa_match = re.search(r'<final_answer>(.*?)</final_answer>', text, re.DOTALL)
    result['final_answer'] = fa_match.group(1).strip()[:200] if fa_match else ""

    # Metacognitive markers
    markers = []
    patterns = [
        (r"[Bb]ut wait", "but wait"),
        (r"[Aa]ctually", "actually"),
        (r"[Ll]et me reconsider", "reconsider"),
        (r"[Hh]mm", "hmm"),
        (r"[Ww]ait,", "wait"),
        (r"[Oo]n second thought", "second thought"),
        (r"[Ll]et me think again", "think again"),
        (r"[Nn]o,? that'?s not", "no that's not"),
        (r"[Ii] was wrong", "I was wrong"),
        (r"[Ll]et me re-", "let me re-"),
        (r"[Hh]old on", "hold on"),
        (r"[Cc]orrection:", "correction"),
    ]
    for p, label in patterns:
        count = len(re.findall(p, text))
        if count > 0:
            markers.extend([label] * count)
    result['metacog_markers'] = markers
    result['metacog_count'] = len(markers)

    # Structure indicators
    struct_count = 0
    for p in [r"[Ss]tep \d", r"^\d+\.\s", r"^\s*[-*]\s", r"[Ff]irst,", r"[Ss]econd,", r"[Tt]herefore,"]:
        if re.search(p, text, re.MULTILINE):
            struct_count += 1
    result['structured'] = struct_count >= 2

    # Post-answer waste
    if result['has_final_answer']:
        after = text.split('</final_answer>')[-1].strip()
        result['post_answer_waste'] = len(after)
    else:
        result['post_answer_waste'] = 0

    # Empty think (model outputs <think>\n</think> then reasoning outside)
    result['empty_think'] = result['has_think'] and result['think_length'] < 10

    return result


def main():
    early_file = RUN_DIR / "train_iteration_000000.html"
    late_file = RUN_DIR / "train_iteration_000018.html"

    print("=" * 80)
    print("PARSING TRACES")
    print("=" * 80)

    early_groups = parse_html_trace(early_file)
    late_groups = parse_html_trace(late_file)
    print(f"Early: {len(early_groups)} problem groups, {sum(len(g.responses) for g in early_groups)} total responses")
    print(f"Late:  {len(late_groups)} problem groups, {sum(len(g.responses) for g in late_groups)} total responses")

    # Analyze all responses
    early_analyses = []
    late_analyses = []

    for g in early_groups:
        for r in g.responses:
            early_analyses.append(analyze_response(r))

    for g in late_groups:
        for r in g.responses:
            late_analyses.append(analyze_response(r))

    # Print detailed comparison
    for label, analyses in [("EARLY (iter 0)", early_analyses), ("LATE (iter 18)", late_analyses)]:
        print(f"\n{'='*80}")
        print(f"  {label}: {len(analyses)} responses")
        print(f"{'='*80}")

        lengths = [a['length'] for a in analyses]
        think_lens = [a['think_length'] for a in analyses]
        post_think_lens = [a['post_think_length'] for a in analyses]
        metacog = [a['metacog_count'] for a in analyses]

        print(f"  Response length:  mean={sum(lengths)/len(lengths):.0f}  median={sorted(lengths)[len(lengths)//2]}  min={min(lengths)}  max={max(lengths)}")
        print(f"  Think length:     mean={sum(think_lens)/len(think_lens):.0f}  min={min(think_lens)}  max={max(think_lens)}")
        print(f"  Post-think len:   mean={sum(post_think_lens)/len(post_think_lens):.0f}")
        print(f"  Has <think>:      {sum(1 for a in analyses if a['has_think'])}/{len(analyses)}")
        print(f"  Empty <think>:    {sum(1 for a in analyses if a['empty_think'])}/{len(analyses)}")
        print(f"  Has final_answer: {sum(1 for a in analyses if a['has_final_answer'])}/{len(analyses)}")
        print(f"  Structured:       {sum(1 for a in analyses if a['structured'])}/{len(analyses)}")
        print(f"  Metacog markers:  mean={sum(metacog)/len(metacog):.1f}  max={max(metacog)}")
        print(f"  Post-answer waste (>50ch): {sum(1 for a in analyses if a['post_answer_waste'] > 50)}/{len(analyses)}")

        # Distribution of metacog markers
        all_markers = []
        for a in analyses:
            all_markers.extend(a['metacog_markers'])
        if all_markers:
            mc = Counter(all_markers).most_common(10)
            print(f"  Top markers: {mc}")

    # Detailed response samples
    print(f"\n{'='*80}")
    print("SAMPLE RESPONSES - EARLY (iter 0)")
    print(f"{'='*80}")

    for gi, g in enumerate(early_groups[:3]):
        print(f"\n--- Problem {gi} (first 200 chars) ---")
        print(f"  {g.problem_text[:200]}")
        print(f"  Reward tables: {len(g.trajectory_rewards)} trajectories")
        for ti, traj in enumerate(g.trajectory_rewards[:3]):
            for tr in traj:
                if tr.step == 'total':
                    print(f"    Trajectory {ti}: total reward={tr.reward}")

        for ri, r in enumerate(g.responses[:3]):
            a = analyze_response(r)
            print(f"\n  Response {ri}: {a['length']} chars, metacog={a['metacog_count']}, structured={a['structured']}")
            print(f"    empty_think={a['empty_think']}, final_answer={a['has_final_answer']}")
            # Show first 250 chars of actual reasoning (after <think>)
            if a['empty_think']:
                # Reasoning is after </think>
                content = r.split('</think>')[-1].strip()[:250]
                print(f"    [EMPTY THINK] Post-think content: {content}")
            elif a['think_content']:
                print(f"    Think content (first 250): {a['think_content'][:250]}")
            else:
                print(f"    Raw response (first 250): {r[:250]}")

    print(f"\n{'='*80}")
    print("SAMPLE RESPONSES - LATE (iter 18)")
    print(f"{'='*80}")

    for gi, g in enumerate(late_groups[:3]):
        print(f"\n--- Problem {gi} (first 200 chars) ---")
        print(f"  {g.problem_text[:200]}")
        print(f"  Reward tables: {len(g.trajectory_rewards)} trajectories")
        for ti, traj in enumerate(g.trajectory_rewards[:3]):
            for tr in traj:
                if tr.step == 'total':
                    print(f"    Trajectory {ti}: total reward={tr.reward}")

        for ri, r in enumerate(g.responses[:3]):
            a = analyze_response(r)
            print(f"\n  Response {ri}: {a['length']} chars, metacog={a['metacog_count']}, structured={a['structured']}")
            print(f"    empty_think={a['empty_think']}, final_answer={a['has_final_answer']}")
            if a['empty_think']:
                content = r.split('</think>')[-1].strip()[:250]
                print(f"    [EMPTY THINK] Post-think content: {content}")
            elif a['think_content']:
                print(f"    Think content (first 250): {a['think_content'][:250]}")
            else:
                print(f"    Raw response (first 250): {r[:250]}")

    # Voice analysis: compare how reasoning starts
    print(f"\n{'='*80}")
    print("VOICE ANALYSIS - How does reasoning BEGIN?")
    print(f"{'='*80}")

    print("\n  EARLY first lines of reasoning (after <think> or start):")
    for i, a in enumerate(early_analyses[:10]):
        if a['empty_think']:
            # Get text after </think>
            g_idx = i // 8
            r_idx = i % 8
            if g_idx < len(early_groups) and r_idx < len(early_groups[g_idx].responses):
                text = early_groups[g_idx].responses[r_idx].split('</think>')[-1].strip()
                first_line = text[:150].split('\n')[0]
                print(f"    [{i}] [EMPTY-THINK] {first_line}")
        elif a['think_content']:
            first_line = a['think_content'][:150].split('\n')[0]
            print(f"    [{i}] [THINK] {first_line}")

    print("\n  LATE first lines of reasoning:")
    for i, a in enumerate(late_analyses[:10]):
        if a['empty_think']:
            g_idx = i // 8
            r_idx = i % 8
            if g_idx < len(late_groups) and r_idx < len(late_groups[g_idx].responses):
                text = late_groups[g_idx].responses[r_idx].split('</think>')[-1].strip()
                first_line = text[:150].split('\n')[0]
                print(f"    [{i}] [EMPTY-THINK] {first_line}")
        elif a['think_content']:
            first_line = a['think_content'][:150].split('\n')[0]
            print(f"    [{i}] [THINK] {first_line}")

    # Format failures deep dive
    print(f"\n{'='*80}")
    print("FORMAT FAILURE DEEP DIVE")
    print(f"{'='*80}")

    for label, groups in [("EARLY", early_groups), ("LATE", late_groups)]:
        print(f"\n  --- {label} ---")
        failure_count = 0
        for gi, g in enumerate(groups):
            for ri, r in enumerate(g.responses):
                a = analyze_response(r)
                if not a['has_final_answer'] and failure_count < 3:
                    failure_count += 1
                    print(f"\n  [Failure {failure_count}] Group {gi}, Response {ri}")
                    print(f"    Problem: {g.problem_text[:150]}...")
                    print(f"    Length: {a['length']} chars")

                    # Why did it fail?
                    last_500 = r[-500:]
                    has_boxed = "\\boxed" in r
                    has_answer_is = bool(re.search(r"[Tt]he answer is", r))
                    has_plain_letter = bool(re.search(r"\b[A-D]\b\s*$", r.strip()))
                    appears_truncated = a['length'] > 15000 and not a['has_final_answer']

                    if appears_truncated:
                        print(f"    DIAGNOSIS: Likely TRUNCATED (long response, no final answer)")
                    elif has_boxed:
                        print(f"    DIAGNOSIS: Used \\boxed instead of <final_answer>")
                    elif has_answer_is:
                        print(f"    DIAGNOSIS: Used prose 'the answer is' instead of tags")
                    elif has_plain_letter:
                        print(f"    DIAGNOSIS: Ended with plain letter answer")
                    else:
                        print(f"    DIAGNOSIS: Never reached answer (spiraling)")

                    print(f"    Last 300 chars: ...{last_500[-300:]}")

    # Check for think usage evolution specifically
    print(f"\n{'='*80}")
    print("THINK TAG EVOLUTION")
    print(f"{'='*80}")
    for label, analyses in [("EARLY", early_analyses), ("LATE", late_analyses)]:
        empty = sum(1 for a in analyses if a['empty_think'])
        has_content = sum(1 for a in analyses if a['has_think'] and not a['empty_think'])
        no_think = sum(1 for a in analyses if not a['has_think'])
        print(f"  {label}: empty_think={empty}  think_with_content={has_content}  no_think={no_think}")

    # Check metrics.jsonl for accuracy progression
    print(f"\n{'='*80}")
    print("METRICS PROGRESSION (from metrics.jsonl)")
    print(f"{'='*80}")
    metrics_file = RUN_DIR / "metrics.jsonl"
    if metrics_file.exists():
        metrics = [json.loads(line) for line in metrics_file.read_text().strip().split('\n')]
        for m in metrics[:5]:
            print(f"  Step {m.get('step', '?')}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != 'step'})}")
        print(f"  ...")
        for m in metrics[-5:]:
            print(f"  Step {m.get('step', '?')}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != 'step'})}")


if __name__ == "__main__":
    main()
