"""Analyze within-group rollout diversity in GPQA RLVR training logs."""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from html.parser import HTMLParser
import difflib

BASE = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/gpqa-experiment")


class LogtreeParser(HTMLParser):
    """Extract episodes from logtree HTML trace files.

    Each episode is a sequence of paragraphs within a section:
      - "Problem: ..."
      - "Response: ..."
      - "Reference Answer: ..."
      - "Boxed: ✓/✗, EOS: ✓/✗, Correct: ✓/✗, Reward: X.XX"
    """

    def __init__(self):
        super().__init__()
        self.in_p = False
        self.current_text = ""
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            cls = dict(attrs).get("class", "")
            if "lt-p" in cls:
                self.in_p = True
                self.current_text = ""

    def handle_endtag(self, tag):
        if tag == "p" and self.in_p:
            self.in_p = False
            self.paragraphs.append(self.current_text.strip())

    def handle_data(self, data):
        if self.in_p:
            self.current_text += data


def parse_episodes(html_path: Path) -> list[dict]:
    """Parse episodes from a logtree HTML file."""
    parser = LogtreeParser()
    parser.feed(html_path.read_text())

    episodes = []
    i = 0
    paras = parser.paragraphs

    while i < len(paras):
        p = paras[i]
        if p.startswith("Problem:"):
            problem = p[len("Problem:"):].strip()
            response = ""
            ref_answer = ""
            correct = None
            reward = None
            boxed = None
            eos = None

            # Look ahead for Response, Reference Answer, verdict
            j = i + 1
            while j < len(paras) and j < i + 5:
                q = paras[j]
                if q.startswith("Response:"):
                    response = q[len("Response:"):].strip()
                elif q.startswith("Reference Answer:"):
                    ref_answer = q[len("Reference Answer:"):].strip()
                elif "Correct:" in q and "Reward:" in q:
                    correct = "Correct: ✓" in q
                    m = re.search(r"Reward:\s*([\d.]+)", q)
                    reward = float(m.group(1)) if m else None
                    boxed = "Boxed: ✓" in q
                    eos = "EOS: ✓" in q
                    j += 1
                    break
                j += 1

            episodes.append({
                "problem": problem,
                "response": response,
                "ref_answer": ref_answer,
                "correct": correct,
                "reward": reward,
                "boxed": boxed,
                "eos": eos,
            })
            i = j
        else:
            i += 1

    return episodes


def group_episodes(episodes: list[dict]) -> list[list[dict]]:
    """Group consecutive episodes with the same problem text."""
    if not episodes:
        return []

    groups = []
    current_group = [episodes[0]]

    for ep in episodes[1:]:
        if ep["problem"] == current_group[0]["problem"]:
            current_group.append(ep)
        else:
            groups.append(current_group)
            current_group = [ep]

    groups.append(current_group)
    return groups


def analyze_group_verdicts(groups: list[list[dict]]) -> dict:
    """Compute verdict diversity statistics for groups."""
    n_all_good = 0
    n_all_bad = 0
    n_mixed = 0
    n_zero_reward_var = 0

    for group in groups:
        rewards = [ep["reward"] for ep in group if ep["reward"] is not None]
        corrects = [ep["correct"] for ep in group if ep["correct"] is not None]

        if not corrects:
            continue

        all_correct = all(corrects)
        all_wrong = not any(corrects)

        if all_correct:
            n_all_good += 1
        elif all_wrong:
            n_all_bad += 1
        else:
            n_mixed += 1

        # Check reward variance
        if rewards and (max(rewards) - min(rewards)) < 1e-9:
            n_zero_reward_var += 1

    total = n_all_good + n_all_bad + n_mixed
    return {
        "total_groups": total,
        "all_good": n_all_good,
        "all_bad": n_all_bad,
        "mixed": n_mixed,
        "frac_all_good": n_all_good / total if total else 0,
        "frac_all_bad": n_all_bad / total if total else 0,
        "frac_mixed": n_mixed / total if total else 0,
        "zero_reward_var": n_zero_reward_var,
        "frac_zero_reward_var": n_zero_reward_var / total if total else 0,
    }


def compare_responses_in_group(group: list[dict], group_idx: int):
    """Character-level comparison of responses within a group."""
    responses = [ep["response"] for ep in group]
    n = len(responses)

    problem_preview = group[0]["problem"][:120] + "..."
    print(f"\n{'='*80}")
    print(f"Group {group_idx}: {n} rollouts | Problem: {problem_preview}")
    print(f"Ref answer: {group[0]['ref_answer']}")

    # Check if all responses are literally identical
    unique_responses = set(responses)
    print(f"Unique responses: {len(unique_responses)} / {n}")

    # Verdicts
    verdicts = ["✓" if ep["correct"] else "✗" for ep in group]
    print(f"Verdicts: {' '.join(verdicts)}")

    # Pairwise similarity
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            ratio = difflib.SequenceMatcher(None, responses[i], responses[j]).ratio()
            similarities.append(ratio)

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        print(f"Pairwise char similarity: avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")

    # Length stats
    lengths = [len(r) for r in responses]
    print(f"Response lengths (chars): {lengths}")

    # Extract final answers
    final_answers = []
    for r in responses:
        m = re.search(r"<final_answer>(.*?)</final_answer>", r, re.DOTALL)
        if m:
            final_answers.append(m.group(1).strip())
        else:
            # Try \\boxed
            m2 = re.search(r"\\boxed\{(.*?)\}", r)
            if m2:
                final_answers.append(m2.group(1).strip())
            else:
                final_answers.append("[NO_ANSWER]")

    unique_answers = set(final_answers)
    print(f"Final answers: {final_answers}")
    print(f"Unique final answers: {len(unique_answers)} / {n}")

    # Classify diversity
    if len(unique_responses) == 1:
        print(">> LITERALLY IDENTICAL responses")
    elif avg_sim > 0.95:
        print(">> Near-identical (cosmetic differences only)")
    elif avg_sim > 0.7:
        print(">> Same structure, different details")
    else:
        print(">> Genuinely different approaches")


def analyze_run(run_dir: Path, group_size: int, label: str):
    """Full analysis for one run."""
    print(f"\n{'#'*80}")
    print(f"# {label} (G={group_size})")
    print(f"# {run_dir}")
    print(f"{'#'*80}")

    # Load metrics.jsonl for ground truth
    metrics_path = run_dir / "metrics.jsonl"
    metrics = []
    if metrics_path.exists():
        for line in metrics_path.read_text().strip().split("\n"):
            metrics.append(json.loads(line))

    # Find trace files
    train_files = sorted(run_dir.glob("train_iteration_*.html"))
    if not train_files:
        print("No trace files found!")
        return

    print(f"\nFound {len(train_files)} trace files")

    # Pick early and late traces
    early_file = train_files[0]
    late_file = train_files[-1]

    for trace_file, phase in [(early_file, "EARLY"), (late_file, "LATE")]:
        iter_match = re.search(r"iteration_(\d+)", trace_file.name)
        iter_num = int(iter_match.group(1)) if iter_match else -1

        print(f"\n{'='*80}")
        print(f"Phase: {phase} (iteration {iter_num}, file: {trace_file.name})")
        print(f"{'='*80}")

        episodes = parse_episodes(trace_file)
        print(f"Total episodes parsed: {len(episodes)}")

        groups = group_episodes(episodes)
        print(f"Total groups: {len(groups)}")
        print(f"Group sizes: {[len(g) for g in groups]}")

        # Verdict analysis
        stats = analyze_group_verdicts(groups)
        print(f"\nVerdict distribution:")
        print(f"  all_good: {stats['all_good']}/{stats['total_groups']} = {stats['frac_all_good']:.4f}")
        print(f"  all_bad:  {stats['all_bad']}/{stats['total_groups']} = {stats['frac_all_bad']:.4f}")
        print(f"  mixed:    {stats['mixed']}/{stats['total_groups']} = {stats['frac_mixed']:.4f}")
        print(f"  zero_reward_var: {stats['zero_reward_var']}/{stats['total_groups']} = {stats['frac_zero_reward_var']:.4f}")

        # Compare with metrics.jsonl
        if iter_num < len(metrics):
            m = metrics[iter_num]
            print(f"\nmetrics.jsonl comparison (step {iter_num}):")
            print(f"  frac_all_good: metrics={m.get('env/all/by_group/frac_all_good', 'N/A'):.4f} vs parsed={stats['frac_all_good']:.4f}")
            print(f"  frac_all_bad:  metrics={m.get('env/all/by_group/frac_all_bad', 'N/A'):.4f} vs parsed={stats['frac_all_bad']:.4f}")
            print(f"  frac_mixed:    metrics={m.get('env/all/by_group/frac_mixed', 'N/A'):.4f} vs parsed={stats['frac_mixed']:.4f}")
            print(f"  entropy:       {m.get('optim/entropy', 'N/A')}")
            print(f"  NOTE: metrics cover ALL groups in batch; trace only logs a subset")

        # Detailed comparison for 3 groups
        print(f"\n--- Detailed response comparison (up to 3 groups) ---")
        for idx, group in enumerate(groups[:3]):
            compare_responses_in_group(group, idx)


def print_metrics_timeseries(run_dir: Path, label: str):
    """Print key metrics over time."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return

    metrics = []
    for line in metrics_path.read_text().strip().split("\n"):
        metrics.append(json.loads(line))

    print(f"\n{'='*80}")
    print(f"Metrics timeseries: {label}")
    print(f"{'='*80}")
    print(f"{'step':>4} {'frac_mixed':>11} {'frac_all_good':>14} {'frac_all_bad':>13} {'entropy':>8} {'correct':>8} {'reward':>8}")

    for m in metrics:
        step = m.get("step", "?")
        fm = m.get("env/all/by_group/frac_mixed", float("nan"))
        fg = m.get("env/all/by_group/frac_all_good", float("nan"))
        fb = m.get("env/all/by_group/frac_all_bad", float("nan"))
        ent = m.get("optim/entropy", float("nan"))
        corr = m.get("env/all/correct", float("nan"))
        rew = m.get("env/all/reward/total", float("nan"))
        print(f"{step:>4} {fm:>11.4f} {fg:>14.4f} {fb:>13.4f} {ent:>8.4f} {corr:>8.4f} {rew:>8.4f}")


if __name__ == "__main__":
    # G=8 run
    g8_dir = BASE / "gpqa-g8-s42"
    analyze_run(g8_dir, group_size=8, label="G=8, seed=42")
    print_metrics_timeseries(g8_dir, "G=8")

    # G=16 run
    g16_dir = BASE / "gpqa-g16-s42"
    analyze_run(g16_dir, group_size=16, label="G=16, seed=42")
    print_metrics_timeseries(g16_dir, "G=16")

    # Summary comparison
    print(f"\n{'#'*80}")
    print("# SUMMARY: G=8 vs G=16 comparison")
    print(f"{'#'*80}")

    for gdir, gs in [(g8_dir, 8), (g16_dir, 16)]:
        metrics_path = gdir / "metrics.jsonl"
        metrics = [json.loads(l) for l in metrics_path.read_text().strip().split("\n")]

        early = metrics[0]
        late = metrics[-1]

        print(f"\nG={gs}:")
        print(f"  Early (step {early['step']}): mixed={early['env/all/by_group/frac_mixed']:.3f}, "
              f"all_good={early['env/all/by_group/frac_all_good']:.3f}, "
              f"all_bad={early['env/all/by_group/frac_all_bad']:.3f}, "
              f"entropy={early.get('optim/entropy', 'N/A')}")
        print(f"  Late  (step {late['step']}): mixed={late['env/all/by_group/frac_mixed']:.3f}, "
              f"all_good={late['env/all/by_group/frac_all_good']:.3f}, "
              f"all_bad={late['env/all/by_group/frac_all_bad']:.3f}, "
              f"entropy={late.get('optim/entropy', 'N/A')}")

        # Average frac with zero reward variance
        total_zero = 0
        total_groups = 0
        for m in metrics:
            fg = m.get("env/all/by_group/frac_all_good", 0)
            fb = m.get("env/all/by_group/frac_all_bad", 0)
            total_zero += (fg + fb)
            total_groups += 1
        avg_zero = total_zero / total_groups if total_groups else 0
        print(f"  Avg frac zero-reward-variance (all_good+all_bad): {avg_zero:.3f}")
