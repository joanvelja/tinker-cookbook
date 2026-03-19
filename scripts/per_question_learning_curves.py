"""Per-question learning curve analysis across GPQA training steps.

DATA LIMITATION: HTML traces only log num_groups_to_log=4 out of 64 problems
per step. So per-question tracking across steps is impossible from traces alone.
We combine: (a) the 95 unique questions from HTML for distribution analysis,
(b) aggregate metrics.jsonl for full-batch trends, (c) cross-seed comparison.
"""

import re
import html
import json
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/gpqa-experiment")


def parse_trace_file(fpath: Path) -> list[dict]:
    """Parse an HTML trace file, extracting per-rollout data."""
    text = fpath.read_text()

    m = re.search(r'iteration_(\d+)', fpath.name)
    step = int(m.group(1)) if m else -1

    problem_pattern = re.compile(r'Problem:\s*(.*?)\s*</p>', re.DOTALL)
    outcome_pattern = re.compile(
        r'Boxed:\s*(✓|✗),\s*EOS:\s*(✓|✗),\s*Correct:\s*(✓|✗),\s*Reward:\s*([-\d.]+)'
    )
    ref_pattern = re.compile(r'Reference Answer:\s*(.*?)\s*</p>', re.DOTALL)

    markers = []
    for m2 in problem_pattern.finditer(text):
        markers.append((m2.start(), 'problem', html.unescape(m2.group(1).strip())[:150]))
    for m2 in ref_pattern.finditer(text):
        markers.append((m2.start(), 'ref', html.unescape(m2.group(1).strip())))
    for m2 in outcome_pattern.finditer(text):
        markers.append((m2.start(), 'outcome', {
            'boxed': m2.group(1) == '✓',
            'correct': m2.group(3) == '✓',
            'reward': float(m2.group(4)),
        }))
    markers.sort(key=lambda x: x[0])

    results = []
    current_problem = None
    current_ref = None

    for _, kind, data in markers:
        if kind == 'problem':
            current_problem = data
        elif kind == 'ref':
            current_ref = data
        elif kind == 'outcome':
            if current_problem:
                results.append({
                    'step': step,
                    'question': current_problem,
                    'reference_answer': current_ref,
                    'boxed': data['boxed'],
                    'correct': data['correct'],
                    'reward': data['reward'],
                })
    return results


def parse_run(run_dir: Path, prefix: str = "train") -> list[dict]:
    """Parse all trace files in a run directory."""
    all_results = []
    for fpath in sorted(run_dir.glob(f"{prefix}_iteration_*.html")):
        all_results.extend(parse_trace_file(fpath))
    return all_results


def load_metrics(run_dir: Path) -> list[dict]:
    """Load metrics.jsonl."""
    metrics = []
    mpath = run_dir / "metrics.jsonl"
    if mpath.exists():
        with open(mpath) as f:
            for line in f:
                metrics.append(json.loads(line))
    return metrics


def analyze_logged_questions(results: list[dict], run_name: str):
    """Analyze the subset of questions logged in HTML traces."""
    by_question = defaultdict(list)
    for r in results:
        by_question[r['question']].append(r)

    print(f"\n{'='*80}")
    print(f"LOGGED QUESTIONS ANALYSIS: {run_name}")
    print(f"(NOTE: only 4 of 64 problems logged per step)")
    print(f"{'='*80}")

    n_unique = len(by_question)
    print(f"\nUnique questions seen across all steps: {n_unique}")
    print(f"Total rollouts: {len(results)}")

    # Per-question accuracy (within-group, single step)
    question_accs = {}
    for q, rollouts in by_question.items():
        # Each question appears in one step, with up to 8 rollouts
        corrects = [r['correct'] for r in rollouts]
        question_accs[q] = sum(corrects) / len(corrects)

    # Categorize by within-group accuracy
    always_correct = sum(1 for a in question_accs.values() if a == 1.0)
    always_incorrect = sum(1 for a in question_accs.values() if a == 0.0)
    mixed = sum(1 for a in question_accs.values() if 0 < a < 1)

    print(f"\nWithin-group accuracy distribution (group_size=8):")
    print(f"  All 8/8 correct:  {always_correct}")
    print(f"  All 0/8 correct:  {always_incorrect}")
    print(f"  Mixed (1-7/8):    {mixed}")

    # Histogram
    print(f"\n  Accuracy histogram:")
    buckets = [(0, 0, "0.000"), (0.001, 0.25, "0.001-0.250"), (0.251, 0.5, "0.251-0.500"),
               (0.501, 0.75, "0.501-0.750"), (0.751, 0.999, "0.751-0.999"), (1.0, 1.0, "1.000")]
    for lo, hi, label in buckets:
        count = sum(1 for a in question_accs.values() if lo <= a <= hi)
        bar = '#' * count
        print(f"    {label:>12}: {count:3d} {bar}")

    # Format rate per step
    format_by_step = defaultdict(list)
    correct_by_step = defaultdict(list)
    for r in results:
        format_by_step[r['step']].append(r['boxed'])
        correct_by_step[r['step']].append(r['correct'])

    print(f"\n  Per-step rates (logged subset only):")
    print(f"  {'Step':>4} | {'Correct':>7} | {'Boxed':>7}")
    for step in sorted(correct_by_step.keys()):
        c_vals = correct_by_step[step]
        f_vals = format_by_step[step]
        print(f"  {step:4d} | {sum(c_vals)/len(c_vals):.3f}   | {sum(f_vals)/len(f_vals):.3f}")

    return question_accs, by_question


def analyze_full_batch_metrics(run_dir: Path, run_name: str):
    """Analyze aggregate metrics from metrics.jsonl (covers all 64 groups)."""
    metrics = load_metrics(run_dir)
    if not metrics:
        print(f"\n  No metrics.jsonl found for {run_name}")
        return

    print(f"\n{'='*80}")
    print(f"FULL BATCH METRICS (all 64 groups): {run_name}")
    print(f"{'='*80}")

    print(f"\n  {'Step':>4} | {'Correct':>7} | {'Boxed':>7} | {'AllGood':>7} | {'Mixed':>7} | {'AllBad':>7} | {'Reward':>7}")
    print(f"  {'-'*70}")

    for m in metrics:
        step = m['step']
        correct = m.get('env/all/correct', 0)
        boxed = m.get('env/all/format_boxed', 0)
        all_good = m.get('env/all/by_group/frac_all_good', 0)
        mixed = m.get('env/all/by_group/frac_mixed', 0)
        all_bad = m.get('env/all/by_group/frac_all_bad', 0)
        reward = m.get('env/all/reward/total', 0)
        print(f"  {step:4d} | {correct:.3f}   | {boxed:.3f}   | {all_good:.3f}   | {mixed:.3f}   | {all_bad:.3f}   | {reward:+.3f}")

    # Also show eval metrics
    print(f"\n  EVAL metrics (test set):")
    print(f"  {'Step':>4} | {'Correct':>7} | {'Boxed':>7} | {'AllGood':>7} | {'Mixed':>7} | {'AllBad':>7}")
    print(f"  {'-'*60}")
    for m in metrics:
        step = m['step']
        if f'test/env/all/correct' not in m:
            continue
        correct = m['test/env/all/correct']
        boxed = m['test/env/all/format_boxed']
        all_good = m['test/env/all/by_group/frac_all_good']
        mixed = m['test/env/all/by_group/frac_mixed']
        all_bad = m['test/env/all/by_group/frac_all_bad']
        print(f"  {step:4d} | {correct:.3f}   | {boxed:.3f}   | {all_good:.3f}   | {mixed:.3f}   | {all_bad:.3f}")

    # Key insight: frac_mixed tells us about per-question variance
    # If training is working, we'd expect frac_all_good to increase,
    # frac_all_bad to decrease, and frac_mixed to initially increase
    # then decrease as the model either learns or gives up on each question.

    print(f"\n  INTERPRETATION:")
    first = metrics[0]
    last = metrics[-1]
    ag_delta = last.get('env/all/by_group/frac_all_good', 0) - first.get('env/all/by_group/frac_all_good', 0)
    ab_delta = last.get('env/all/by_group/frac_all_bad', 0) - first.get('env/all/by_group/frac_all_bad', 0)
    mx_delta = last.get('env/all/by_group/frac_mixed', 0) - first.get('env/all/by_group/frac_mixed', 0)
    cr_delta = last.get('env/all/correct', 0) - first.get('env/all/correct', 0)

    print(f"  Correct:  {first.get('env/all/correct', 0):.3f} -> {last.get('env/all/correct', 0):.3f} (delta={cr_delta:+.3f})")
    print(f"  AllGood:  {first.get('env/all/by_group/frac_all_good', 0):.3f} -> {last.get('env/all/by_group/frac_all_good', 0):.3f} (delta={ag_delta:+.3f})")
    print(f"  AllBad:   {first.get('env/all/by_group/frac_all_bad', 0):.3f} -> {last.get('env/all/by_group/frac_all_bad', 0):.3f} (delta={ab_delta:+.3f})")
    print(f"  Mixed:    {first.get('env/all/by_group/frac_mixed', 0):.3f} -> {last.get('env/all/by_group/frac_mixed', 0):.3f} (delta={mx_delta:+.3f})")

    # Check if the "mixed" fraction tells us about learning
    # High mixed = model is uncertain. If mixed stays high, model isn't consolidating.
    # If all_good grows at expense of mixed, model is learning specific questions.
    # If all_bad grows at expense of mixed, model is forgetting.


def cross_seed_logged_comparison(run_dirs: dict[str, Path]):
    """Compare logged questions across seeds."""
    print(f"\n{'='*80}")
    print(f"CROSS-SEED COMPARISON (logged questions)")
    print(f"{'='*80}")

    all_questions = {}
    for name, run_dir in run_dirs.items():
        results = parse_run(run_dir)
        by_q = defaultdict(list)
        for r in results:
            by_q[r['question']].append(r['correct'])
        accs = {}
        for q, corrects in by_q.items():
            accs[q] = sum(corrects) / len(corrects)
        all_questions[name] = accs
        print(f"  {name}: {len(accs)} unique questions logged")

    # Find questions that appear in multiple seeds
    all_qs = set()
    for accs in all_questions.values():
        all_qs.update(accs.keys())

    shared = []
    for q in all_qs:
        seeds_with_q = [(name, accs[q]) for name, accs in all_questions.items() if q in accs]
        if len(seeds_with_q) > 1:
            shared.append((q, seeds_with_q))

    print(f"\n  Questions appearing in 2+ seeds: {len(shared)}")
    for q, seeds in shared[:10]:
        seed_str = " | ".join(f"{name}={acc:.2f}" for name, acc in seeds)
        print(f"    {q[:60]}... : {seed_str}")


def cross_seed_metrics_comparison(run_dirs: dict[str, Path]):
    """Compare aggregate metrics across seeds."""
    print(f"\n{'='*80}")
    print(f"CROSS-SEED METRICS COMPARISON")
    print(f"{'='*80}")

    all_metrics = {}
    for name, run_dir in run_dirs.items():
        all_metrics[name] = load_metrics(run_dir)

    # Compare step 0 accuracy and group fractions
    print(f"\n  Step 0 comparison:")
    print(f"  {'Run':>12} | {'Correct':>7} | {'Boxed':>7} | {'AllGood':>7} | {'Mixed':>7} | {'AllBad':>7}")
    print(f"  {'-'*65}")
    for name, metrics in all_metrics.items():
        if not metrics:
            continue
        m = metrics[0]
        print(f"  {name:>12} | {m.get('env/all/correct',0):.3f}   | {m.get('env/all/format_boxed',0):.3f}   | {m.get('env/all/by_group/frac_all_good',0):.3f}   | {m.get('env/all/by_group/frac_mixed',0):.3f}   | {m.get('env/all/by_group/frac_all_bad',0):.3f}")

    # Compare final step
    print(f"\n  Final step comparison:")
    print(f"  {'Run':>12} | {'Step':>4} | {'Correct':>7} | {'Boxed':>7} | {'AllGood':>7} | {'Mixed':>7} | {'AllBad':>7}")
    print(f"  {'-'*72}")
    for name, metrics in all_metrics.items():
        if not metrics:
            continue
        m = metrics[-1]
        print(f"  {name:>12} | {m['step']:4d} | {m.get('env/all/correct',0):.3f}   | {m.get('env/all/format_boxed',0):.3f}   | {m.get('env/all/by_group/frac_all_good',0):.3f}   | {m.get('env/all/by_group/frac_mixed',0):.3f}   | {m.get('env/all/by_group/frac_all_bad',0):.3f}")

    # Test set step 0 comparison (same questions across seeds? seeds differ, so questions differ)
    print(f"\n  Eval (test set) at step 0:")
    print(f"  {'Run':>12} | {'Correct':>7} | {'N_episodes':>10}")
    print(f"  {'-'*40}")
    for name, metrics in all_metrics.items():
        if not metrics:
            continue
        m = metrics[0]
        if 'test/env/all/correct' in m:
            print(f"  {name:>12} | {m['test/env/all/correct']:.3f}   | {m['test/env/all/total_episodes']}")


def per_question_variance_analysis(run_dirs: dict[str, Path]):
    """Analyze whether per-question behavior is consistent across seeds via metrics."""
    print(f"\n{'='*80}")
    print(f"PER-QUESTION LEARNING SIGNAL ANALYSIS")
    print(f"{'='*80}")

    for name, run_dir in run_dirs.items():
        metrics = load_metrics(run_dir)
        if not metrics:
            continue

        # Track frac_mixed over time — this is our proxy for per-question learning
        # frac_mixed = fraction of groups where some rollouts are correct, some aren't
        # If model learns: frac_mixed should decrease as groups consolidate to all_good or all_bad
        steps = [m['step'] for m in metrics]
        mixed = [m.get('env/all/by_group/frac_mixed', 0) for m in metrics]
        all_good = [m.get('env/all/by_group/frac_all_good', 0) for m in metrics]
        all_bad = [m.get('env/all/by_group/frac_all_bad', 0) for m in metrics]
        correct = [m.get('env/all/correct', 0) for m in metrics]

        # Correlation between step and frac_all_good (are more questions becoming solidly correct?)
        n = len(steps)
        if n < 3:
            continue

        def slope(xs, ys):
            mx = sum(xs) / n
            my = sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            return num / den if den > 0 else 0

        s_correct = slope(steps, correct)
        s_good = slope(steps, all_good)
        s_bad = slope(steps, all_bad)
        s_mixed = slope(steps, mixed)

        print(f"\n  {name}:")
        print(f"    Slope(correct):  {s_correct:+.5f} per step")
        print(f"    Slope(all_good): {s_good:+.5f} per step")
        print(f"    Slope(all_bad):  {s_bad:+.5f} per step")
        print(f"    Slope(mixed):    {s_mixed:+.5f} per step")

        # Check: at step 0, how many groups are all_good?
        # If this is high and doesn't grow, the model already knows most of what it can know.
        m0 = metrics[0]
        n_groups = int(m0.get('env/all/total_episodes', 512) / 8)  # 64 groups
        est_all_good = int(all_good[0] * n_groups)
        est_all_bad = int(all_bad[0] * n_groups)
        est_mixed = int(mixed[0] * n_groups)
        print(f"    Step 0 estimated: ~{est_all_good} always-correct, ~{est_mixed} mixed, ~{est_all_bad} always-wrong (of {n_groups} groups)")


def main():
    # Analyze all runs
    g8_runs = {
        "gpqa-g8-s42": BASE / "gpqa-g8-s42",
        "gpqa-g8-s137": BASE / "gpqa-g8-s137",
        "gpqa-g8-s7": BASE / "gpqa-g8-s7",
    }
    g16_runs = {
        "gpqa-g16-s42": BASE / "gpqa-g16-s42",
        "gpqa-g16-s137": BASE / "gpqa-g16-s137",
        "gpqa-g16-s7": BASE / "gpqa-g16-s7",
    }

    # 1. Logged questions analysis (HTML traces)
    for name, run_dir in g8_runs.items():
        results = parse_run(run_dir)
        analyze_logged_questions(results, name)

    # 2. Full batch metrics (metrics.jsonl)
    for name, run_dir in g8_runs.items():
        analyze_full_batch_metrics(run_dir, name)

    # 3. Cross-seed comparisons
    cross_seed_logged_comparison(g8_runs)
    cross_seed_metrics_comparison(g8_runs)
    cross_seed_metrics_comparison(g16_runs)

    # 4. Per-question learning signal
    per_question_variance_analysis({**g8_runs, **g16_runs})

    # 5. Key question: is frac_mixed telling us about per-question learning?
    print(f"\n{'='*80}")
    print(f"SUMMARY: IS THE MODEL LEARNING PER-QUESTION?")
    print(f"{'='*80}")
    print("""
    Data limitation: HTML traces log only 4/64 groups per step (num_groups_to_log=4).
    No question repeats across steps in the logged subset.

    Proxy signals from full-batch metrics:
    - frac_all_good: fraction of 64 groups where ALL 8 rollouts are correct
    - frac_mixed:    fraction where SOME rollouts are correct
    - frac_all_bad:  fraction where NO rollouts are correct

    If the model learns specific questions:
      -> frac_all_good should increase over training
      -> frac_all_bad should decrease
      -> frac_mixed serves as the "learning frontier"

    If the model just shifts aggregate probability without per-question learning:
      -> frac_mixed stays high and dominant
      -> frac_all_good and frac_all_bad stay similar
    """)


if __name__ == "__main__":
    main()
