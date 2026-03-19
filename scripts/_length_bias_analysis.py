"""
Investigate PPO length-bias hypothesis in GPQA RLVR training.

Hypothesis: PPO systematically pushes toward longer responses because correct
answers tend to be longer than wrong ones. This increases truncation, degrading
performance.

We test this with actual data from training HTML trace logs + metrics.jsonl.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LOG_DIR = Path("logs/gpqa-experiment/gpqa-g8-s42")

# ---------------------------------------------------------------------------
# 1. Parse HTML trace files for per-episode data
# ---------------------------------------------------------------------------

# Each episode in the HTML has a pattern:
# "Response: ..." followed by "Boxed: ..., EOS: ..., Correct: ..., Reward: ..."
REWARD_LINE_RE = re.compile(
    r"Boxed:\s*(✓|✗),\s*EOS:\s*(✓|✗),\s*Correct:\s*(✓|✗),\s*Reward:\s*([-\d.]+)"
)

# Response text is between "Response: " and the next "Reference Answer:" or next section
RESPONSE_RE = re.compile(
    r"Response:\s*(.*?)\s*(?:</p>)",
    re.DOTALL,
)


def parse_html_trace(path: Path) -> list[dict]:
    """Extract per-episode data from a training HTML trace file."""
    text = path.read_text()

    # Find all response blocks and reward lines
    episodes = []

    # Split by "Problem:" to get individual episodes
    # Each episode has: Problem -> Response -> Reference Answer -> Boxed/EOS/Correct/Reward
    blocks = text.split("Response: ")

    for i, block in enumerate(blocks[1:]):  # skip first split (before any response)
        # Extract response text (everything up to </p>)
        p_end = block.find("</p>")
        if p_end == -1:
            continue
        response_text = block[:p_end].strip()

        # Find the reward line in the remainder
        remainder = block[p_end:]
        m = REWARD_LINE_RE.search(remainder)
        if not m:
            continue

        boxed = m.group(1) == "✓"
        eos = m.group(2) == "✓"
        correct = m.group(3) == "✓"
        reward = float(m.group(4))

        # Decode HTML entities for length measurement
        response_clean = (
            response_text
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&#x27;", "'")
            .replace("&quot;", '"')
        )

        episodes.append({
            "response_chars": len(response_clean),
            "boxed": boxed,
            "eos": eos,
            "correct": correct,
            "reward": reward,
            "truncated": not eos,
        })

    return episodes


def categorize_episode(ep: dict) -> str:
    """Categorize into reward buckets."""
    if ep["correct"]:
        return "correct"
    elif ep["truncated"]:
        return "truncated"
    elif ep["boxed"]:
        return "wrong_formatted"
    else:
        return "wrong_unformatted"


# ---------------------------------------------------------------------------
# 2. Parse metrics.jsonl for time-series
# ---------------------------------------------------------------------------

def load_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().strip().split("\n"):
        rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PPO LENGTH-BIAS HYPOTHESIS INVESTIGATION")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Part 1: Per-episode length distributions by reward bucket
    # -----------------------------------------------------------------------
    print("\n\n### PART 1: Length distributions by reward category ###\n")

    all_episodes = []
    episodes_by_step = {}

    for html_path in sorted(LOG_DIR.glob("train_iteration_*.html")):
        step = int(html_path.stem.split("_")[-1])
        eps = parse_html_trace(html_path)
        episodes_by_step[step] = eps
        for ep in eps:
            ep["step"] = step
        all_episodes.extend(eps)

    print(f"Total episodes parsed from HTML traces: {len(all_episodes)}")
    print(f"Steps with trace data: {sorted(episodes_by_step.keys())}")

    # Note: HTML traces only log a subset (num_groups_to_log=4 from config,
    # so 4 groups * 8 episodes = 32 per step). This is a sample, not the full batch.

    # Aggregate by category
    by_cat = defaultdict(list)
    for ep in all_episodes:
        cat = categorize_episode(ep)
        by_cat[cat].append(ep["response_chars"])

    print(f"\n{'Category':<20} {'Count':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 78)
    for cat in ["correct", "wrong_formatted", "wrong_unformatted", "truncated"]:
        lengths = by_cat.get(cat, [])
        if not lengths:
            print(f"{cat:<20} {'0':>6}")
            continue
        arr = np.array(lengths)
        print(f"{cat:<20} {len(arr):>6} {arr.mean():>8.0f} {np.median(arr):>8.0f} {arr.std():>8.0f} {arr.min():>8} {arr.max():>8}")

    # Statistical test: are correct responses longer than wrong ones?
    correct_lens = np.array(by_cat.get("correct", []))
    wrong_fmt_lens = np.array(by_cat.get("wrong_formatted", []))

    if len(correct_lens) > 0 and len(wrong_fmt_lens) > 0:
        diff = correct_lens.mean() - wrong_fmt_lens.mean()
        ratio = correct_lens.mean() / wrong_fmt_lens.mean() if wrong_fmt_lens.mean() > 0 else float("inf")
        print(f"\nCorrect vs Wrong-formatted mean difference: {diff:.0f} chars")
        print(f"Correct/Wrong-formatted ratio: {ratio:.2f}x")

        if diff > 0:
            print(">>> CONSISTENT with hypothesis: correct responses are longer")
        else:
            print(">>> INCONSISTENT with hypothesis: correct responses are NOT longer")

    # -----------------------------------------------------------------------
    # Part 2: Length and correctness trends over training (from metrics.jsonl)
    # -----------------------------------------------------------------------
    print("\n\n### PART 2: ac_tokens_per_turn vs correct rate over training ###\n")

    metrics = load_metrics(LOG_DIR / "metrics.jsonl")

    steps = []
    ac_tokens = []
    correct_rates = []
    entropies = []

    for m in metrics:
        steps.append(m["step"])
        ac_tokens.append(m["env/all/ac_tokens_per_turn"])
        correct_rates.append(m["env/all/correct"])
        entropies.append(m.get("optim/entropy", float("nan")))

    steps = np.array(steps)
    ac_tokens = np.array(ac_tokens)
    correct_rates = np.array(correct_rates)
    entropies = np.array(entropies)

    # Correlation: ac_tokens vs correct_rate
    corr_len_correct = np.corrcoef(ac_tokens, correct_rates)[0, 1]
    print(f"Pearson correlation (ac_tokens_per_turn, correct_rate): {corr_len_correct:.3f}")

    # Correlation: ac_tokens vs step (is length increasing?)
    corr_len_step = np.corrcoef(steps, ac_tokens)[0, 1]
    print(f"Pearson correlation (step, ac_tokens_per_turn): {corr_len_step:.3f}")

    # Correlation: correct vs step (is accuracy decreasing?)
    corr_correct_step = np.corrcoef(steps, correct_rates)[0, 1]
    print(f"Pearson correlation (step, correct_rate): {corr_correct_step:.3f}")

    print(f"\nStep-by-step data:")
    print(f"{'Step':>4} {'ac_tokens':>10} {'correct':>8} {'entropy':>8}")
    print("-" * 34)
    for i in range(len(steps)):
        print(f"{steps[i]:>4} {ac_tokens[i]:>10.0f} {correct_rates[i]:>8.3f} {entropies[i]:>8.4f}")

    # First half vs second half
    mid = len(steps) // 2
    print(f"\nFirst half  (steps 0-{steps[mid-1]:<2d}): mean_tokens={ac_tokens[:mid].mean():.0f}, mean_correct={correct_rates[:mid].mean():.3f}")
    print(f"Second half (steps {steps[mid]:<2d}-{steps[-1]:<2d}): mean_tokens={ac_tokens[mid:].mean():.0f}, mean_correct={correct_rates[mid:].mean():.3f}")

    if corr_len_step > 0.3:
        print("\n>>> Length IS increasing over training")
    elif corr_len_step < -0.3:
        print("\n>>> Length is DECREASING over training")
    else:
        print("\n>>> Length has NO clear trend over training")

    if corr_len_correct < -0.3:
        print(">>> Longer responses ARE associated with lower accuracy (consistent with hypothesis)")
    elif corr_len_correct > 0.3:
        print(">>> Longer responses are associated with HIGHER accuracy (inconsistent)")
    else:
        print(">>> No clear relationship between length and accuracy")

    # -----------------------------------------------------------------------
    # Part 3: Test eval set trends (less noisy since eval uses fixed questions)
    # -----------------------------------------------------------------------
    print("\n\n### PART 3: Test eval trends ###\n")

    test_steps = []
    test_ac_tokens = []
    test_correct = []

    for m in metrics:
        if "test/env/all/ac_tokens_per_turn" in m:
            test_steps.append(m["step"])
            test_ac_tokens.append(m["test/env/all/ac_tokens_per_turn"])
            test_correct.append(m["test/env/all/correct"])

    print(f"{'Step':>4} {'test_ac_tokens':>14} {'test_correct':>12}")
    print("-" * 34)
    for i in range(len(test_steps)):
        print(f"{test_steps[i]:>4} {test_ac_tokens[i]:>14.0f} {test_correct[i]:>12.3f}")

    if len(test_steps) > 2:
        test_steps_arr = np.array(test_steps)
        test_ac_arr = np.array(test_ac_tokens)
        test_corr_arr = np.array(test_correct)
        corr_test = np.corrcoef(test_ac_arr, test_corr_arr)[0, 1]
        print(f"\nPearson correlation (test_ac_tokens, test_correct): {corr_test:.3f}")
        corr_test_step = np.corrcoef(test_steps_arr, test_ac_arr)[0, 1]
        print(f"Pearson correlation (step, test_ac_tokens): {corr_test_step:.3f}")

    # -----------------------------------------------------------------------
    # Part 4: Entropy vs length relationship
    # -----------------------------------------------------------------------
    print("\n\n### PART 4: Entropy vs length relationship ###\n")

    corr_entropy_len = np.corrcoef(entropies, ac_tokens)[0, 1]
    corr_entropy_step = np.corrcoef(steps, entropies)[0, 1]

    print(f"Pearson correlation (entropy, ac_tokens_per_turn): {corr_entropy_len:.3f}")
    print(f"Pearson correlation (step, entropy): {corr_entropy_step:.3f}")
    print(f"Entropy range: {entropies.min():.4f} - {entropies.max():.4f}")
    print(f"Entropy first half mean: {entropies[:mid].mean():.4f}")
    print(f"Entropy second half mean: {entropies[mid:].mean():.4f}")

    if abs(corr_entropy_len) > 0.5:
        print(f"\n>>> Strong entropy-length correlation ({corr_entropy_len:.3f}): entropy {'increases' if corr_entropy_len > 0 else 'decreases'} with length")
    else:
        print(f"\n>>> Weak entropy-length correlation ({corr_entropy_len:.3f})")

    # -----------------------------------------------------------------------
    # Part 5: Within-step analysis — in mixed groups, are longer responses
    #          the ones with positive advantages?
    # -----------------------------------------------------------------------
    print("\n\n### PART 5: Within-step length-reward correlation (mixed groups proxy) ###\n")

    # For each step, compute correlation between response length and reward
    # among the sampled episodes. This is a proxy for the advantage direction.
    print(f"{'Step':>4} {'N':>4} {'corr(len,reward)':>16} {'mean_correct_len':>16} {'mean_wrong_len':>15}")
    print("-" * 60)

    step_correlations = []
    for step in sorted(episodes_by_step.keys()):
        eps = episodes_by_step[step]
        if len(eps) < 3:
            continue

        lens = np.array([e["response_chars"] for e in eps])
        rewards = np.array([e["reward"] for e in eps])

        # Only compute correlation if there's variance in both
        if lens.std() < 1 or rewards.std() < 0.001:
            continue

        corr = np.corrcoef(lens, rewards)[0, 1]
        step_correlations.append(corr)

        correct_lens_step = [e["response_chars"] for e in eps if e["correct"]]
        wrong_lens_step = [e["response_chars"] for e in eps if not e["correct"] and e["eos"]]

        mc = np.mean(correct_lens_step) if correct_lens_step else float("nan")
        mw = np.mean(wrong_lens_step) if wrong_lens_step else float("nan")

        print(f"{step:>4} {len(eps):>4} {corr:>16.3f} {mc:>16.0f} {mw:>15.0f}")

    if step_correlations:
        mean_corr = np.mean(step_correlations)
        print(f"\nMean within-step correlation(length, reward): {mean_corr:.3f}")
        if mean_corr > 0.2:
            print(">>> Positive: longer responses tend to get higher rewards (supports hypothesis)")
        elif mean_corr < -0.2:
            print(">>> Negative: shorter responses tend to get higher rewards (contradicts hypothesis)")
        else:
            print(">>> Weak: no clear within-step length-reward relationship")

    # -----------------------------------------------------------------------
    # Part 6: Truncation rate over training
    # -----------------------------------------------------------------------
    print("\n\n### PART 6: Truncation and format rates over training ###\n")

    print(f"{'Step':>4} {'format_boxed':>12} {'format_eos':>10} {'trunc_rate':>10}")
    print("-" * 40)
    for m in metrics:
        eos_rate = m["env/all/format_eos"]
        trunc_rate = 1 - eos_rate
        print(f"{m['step']:>4} {m['env/all/format_boxed']:>12.3f} {eos_rate:>10.3f} {trunc_rate:>10.3f}")

    eos_rates = np.array([m["env/all/format_eos"] for m in metrics])
    corr_eos_step = np.corrcoef(steps, eos_rates)[0, 1]
    print(f"\nPearson correlation (step, eos_rate): {corr_eos_step:.3f}")
    if corr_eos_step < -0.3:
        print(">>> EOS rate is DECREASING (truncation increasing) — consistent with hypothesis")
    elif corr_eos_step > 0.3:
        print(">>> EOS rate is INCREASING (truncation decreasing) — inconsistent")
    else:
        print(">>> No clear truncation trend")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    findings = []

    # Finding 1: Are correct responses longer?
    if len(correct_lens) > 0 and len(wrong_fmt_lens) > 0:
        diff = correct_lens.mean() - wrong_fmt_lens.mean()
        if diff > 100:
            findings.append(f"YES: Correct responses are ~{diff:.0f} chars longer than wrong-formatted ones")
        elif diff < -100:
            findings.append(f"NO: Correct responses are ~{abs(diff):.0f} chars SHORTER than wrong-formatted ones")
        else:
            findings.append(f"UNCLEAR: Correct vs wrong-formatted length difference is small ({diff:.0f} chars)")

    # Finding 2: Is length increasing?
    if corr_len_step > 0.3:
        findings.append(f"YES: Response length is increasing over training (r={corr_len_step:.2f})")
    elif corr_len_step < -0.3:
        findings.append(f"NO: Response length is DECREASING over training (r={corr_len_step:.2f})")
    else:
        findings.append(f"UNCLEAR: No clear length trend over training (r={corr_len_step:.2f})")

    # Finding 3: Is accuracy decreasing?
    if corr_correct_step < -0.3:
        findings.append(f"YES: Accuracy is decreasing over training (r={corr_correct_step:.2f})")
    elif corr_correct_step > 0.3:
        findings.append(f"NO: Accuracy is INCREASING over training (r={corr_correct_step:.2f})")
    else:
        findings.append(f"UNCLEAR: No clear accuracy trend over training (r={corr_correct_step:.2f})")

    # Finding 4: Within-step correlation
    if step_correlations:
        mean_corr = np.mean(step_correlations)
        findings.append(f"Within-step length-reward correlation: {mean_corr:.3f}")

    # Finding 5: Entropy
    findings.append(f"Entropy-length correlation: {corr_entropy_len:.3f}")
    findings.append(f"Entropy trend: {corr_entropy_step:.3f}")

    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")

    # Verdict
    print("\n--- VERDICT ---")

    # The hypothesis requires ALL of:
    # a) correct responses longer than wrong ones
    # b) length increasing over training
    # c) accuracy decreasing over training
    # d) positive within-step length-reward correlation

    supports = 0
    total = 0

    if len(correct_lens) > 0 and len(wrong_fmt_lens) > 0:
        total += 1
        if correct_lens.mean() - wrong_fmt_lens.mean() > 100:
            supports += 1

    total += 1
    if corr_len_step > 0.3:
        supports += 1

    total += 1
    if corr_correct_step < -0.3:
        supports += 1

    if step_correlations:
        total += 1
        if np.mean(step_correlations) > 0.2:
            supports += 1

    print(f"Evidence supporting hypothesis: {supports}/{total} conditions met")

    if supports >= 3:
        print("HYPOTHESIS: LIKELY SUPPORTED")
    elif supports <= 1:
        print("HYPOTHESIS: LIKELY FALSE")
    else:
        print("HYPOTHESIS: MIXED EVIDENCE — needs more investigation")


if __name__ == "__main__":
    main()
