"""
Follow-up analysis: the primary hypothesis (PPO pushes toward longer correct
responses) is FALSIFIED. Correct responses are SHORTER than wrong ones.

But the data shows:
1. Strong negative within-step correlation: length vs reward = -0.49
   (shorter = more likely correct)
2. Yet length IS increasing over training (r=0.41)
3. Truncation rate is increasing (r=-0.58 for EOS rate)

This is paradoxical. PPO should push toward shorter responses (since those
are more rewarded), yet the model gets longer. Let's investigate WHY.

Possible explanations:
A) Truncated responses dominate the gradient. The massive penalty (-0.2)
   for truncation means truncated responses (25k chars) have negative
   advantage. PPO DECREASES their probability. But what replaces them?
   If the model learns to avoid one type of verbosity but substitutes
   another kind, net length could still increase.

B) The advantage is dominated by the correct/wrong signal within non-truncated
   responses. Since correct non-truncated are SHORTER, PPO should push toward
   shorter non-truncated responses. But the format_boxed rate is also dropping,
   meaning more responses fail to format correctly even when not truncated.

C) MaxRL advantage subgroups: with use_advantage_subgroups=True and
   advantage_alpha=0.5, the advantage computation splits groups. This could
   create weird incentives if the subgroup structure interacts with length.

Let's quantify these.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LOG_DIR = Path("logs/gpqa-experiment/gpqa-g8-s42")

REWARD_LINE_RE = re.compile(
    r"Boxed:\s*(✓|✗),\s*EOS:\s*(✓|✗),\s*Correct:\s*(✓|✗),\s*Reward:\s*([-\d.]+)"
)


def parse_html_trace(path: Path) -> list[dict]:
    text = path.read_text()
    episodes = []
    blocks = text.split("Response: ")
    for block in blocks[1:]:
        p_end = block.find("</p>")
        if p_end == -1:
            continue
        response_text = block[:p_end].strip()
        remainder = block[p_end:]
        m = REWARD_LINE_RE.search(remainder)
        if not m:
            continue
        boxed = m.group(1) == "✓"
        eos = m.group(2) == "✓"
        correct = m.group(3) == "✓"
        reward = float(m.group(4))
        response_clean = (
            response_text
            .replace("&lt;", "<").replace("&gt;", ">")
            .replace("&amp;", "&").replace("&#x27;", "'").replace("&quot;", '"')
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


def load_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().strip().split("\n")]


def main():
    print("=" * 80)
    print("FOLLOW-UP: WHY IS LENGTH INCREASING IF SHORTER = MORE CORRECT?")
    print("=" * 80)

    episodes_by_step = {}
    all_episodes = []
    for html_path in sorted(LOG_DIR.glob("train_iteration_*.html")):
        step = int(html_path.stem.split("_")[-1])
        eps = parse_html_trace(html_path)
        episodes_by_step[step] = eps
        for ep in eps:
            ep["step"] = step
        all_episodes.extend(eps)

    # -----------------------------------------------------------------------
    # A) Reward distribution and gradient budget
    # -----------------------------------------------------------------------
    print("\n### A: Reward distribution and what drives the gradient ###\n")

    # Under MaxRL with subgroups, advantage = reward - baseline
    # The baseline is computed per-subgroup. With advantage_alpha=0.5:
    #   subgroup split: top 50% rewards vs bottom 50% rewards
    #   advantage for top: reward - mean(top)
    #   advantage for bottom: reward - mean(bottom)
    # This means the gradient is driven by WITHIN-SUBGROUP differences.
    # If all truncated responses are in the "bottom" subgroup, their
    # advantages are centered around -0.2 (all similar penalty), so
    # the gradient signal from them is weak.

    # Let's look at the reward distribution
    rewards = np.array([e["reward"] for e in all_episodes])
    unique_rewards = np.unique(rewards)
    print(f"Unique reward values: {sorted(unique_rewards)}")
    for r in sorted(unique_rewards):
        count = np.sum(rewards == r)
        pct = 100 * count / len(rewards)
        print(f"  reward={r:>5.2f}: {count:>4} episodes ({pct:.1f}%)")

    # With group_size=8, compute what a typical group looks like
    # A "mixed" group has at least one correct and one wrong.
    # frac_mixed hovers around 0.5-0.7 in the metrics.

    print("\n### B: Non-truncated analysis only ###\n")
    # Filter to non-truncated only
    non_trunc = [e for e in all_episodes if e["eos"]]
    trunc = [e for e in all_episodes if not e["eos"]]

    print(f"Non-truncated: {len(non_trunc)} ({100*len(non_trunc)/len(all_episodes):.1f}%)")
    print(f"Truncated: {len(trunc)} ({100*len(trunc)/len(all_episodes):.1f}%)")

    nt_correct = [e for e in non_trunc if e["correct"]]
    nt_wrong = [e for e in non_trunc if not e["correct"]]

    if nt_correct and nt_wrong:
        c_lens = np.array([e["response_chars"] for e in nt_correct])
        w_lens = np.array([e["response_chars"] for e in nt_wrong])
        print(f"\nNon-truncated correct:  N={len(c_lens)}, mean={c_lens.mean():.0f}, median={np.median(c_lens):.0f}")
        print(f"Non-truncated wrong:    N={len(w_lens)}, mean={w_lens.mean():.0f}, median={np.median(w_lens):.0f}")
        diff = c_lens.mean() - w_lens.mean()
        print(f"Difference (correct - wrong): {diff:.0f} chars")

        if diff > 0:
            print(">>> Among non-truncated: correct are LONGER")
        else:
            print(">>> Among non-truncated: correct are SHORTER")

    # -----------------------------------------------------------------------
    # C) Per-step: what fraction of gradient budget comes from truncated vs non-truncated?
    # -----------------------------------------------------------------------
    print("\n### C: Per-step truncation breakdown ###\n")

    print(f"{'Step':>4} {'N_total':>7} {'N_trunc':>7} {'N_nontrunc':>10} {'trunc%':>7} "
          f"{'corr_len_rew_nontrunc':>21} {'corr_len_rew_all':>16}")
    print("-" * 80)

    for step in sorted(episodes_by_step.keys()):
        eps = episodes_by_step[step]
        n_total = len(eps)
        n_trunc = sum(1 for e in eps if e["truncated"])
        n_nontrunc = n_total - n_trunc

        # Correlation among non-truncated only
        nt_eps = [e for e in eps if not e["truncated"]]
        if len(nt_eps) >= 3:
            nt_lens = np.array([e["response_chars"] for e in nt_eps])
            nt_rews = np.array([e["reward"] for e in nt_eps])
            if nt_lens.std() > 1 and nt_rews.std() > 0.001:
                corr_nt = np.corrcoef(nt_lens, nt_rews)[0, 1]
            else:
                corr_nt = float("nan")
        else:
            corr_nt = float("nan")

        all_lens = np.array([e["response_chars"] for e in eps])
        all_rews = np.array([e["reward"] for e in eps])
        if all_lens.std() > 1 and all_rews.std() > 0.001:
            corr_all = np.corrcoef(all_lens, all_rews)[0, 1]
        else:
            corr_all = float("nan")

        print(f"{step:>4} {n_total:>7} {n_trunc:>7} {n_nontrunc:>10} "
              f"{100*n_trunc/n_total:>6.1f}% {corr_nt:>21.3f} {corr_all:>16.3f}")

    # -----------------------------------------------------------------------
    # D) The key question: among NON-TRUNCATED responses, is correct longer
    #    or shorter? This determines what the non-truncation gradient does.
    # -----------------------------------------------------------------------
    print("\n### D: Non-truncated correct vs wrong per step ###\n")

    print(f"{'Step':>4} {'N_corr':>6} {'N_wrong':>7} {'mean_corr':>9} {'mean_wrong':>10} {'diff':>8}")
    print("-" * 50)

    diffs = []
    for step in sorted(episodes_by_step.keys()):
        eps = episodes_by_step[step]
        nt_corr = [e for e in eps if e["correct"] and not e["truncated"]]
        nt_wrong = [e for e in eps if not e["correct"] and not e["truncated"]]

        if nt_corr and nt_wrong:
            mc = np.mean([e["response_chars"] for e in nt_corr])
            mw = np.mean([e["response_chars"] for e in nt_wrong])
            d = mc - mw
            diffs.append(d)
            print(f"{step:>4} {len(nt_corr):>6} {len(nt_wrong):>7} {mc:>9.0f} {mw:>10.0f} {d:>+8.0f}")
        else:
            print(f"{step:>4} {len(nt_corr):>6} {len(nt_wrong):>7} {'N/A':>9} {'N/A':>10} {'N/A':>8}")

    if diffs:
        mean_diff = np.mean(diffs)
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d <= 0)
        print(f"\nMean diff (correct - wrong among non-truncated): {mean_diff:.0f}")
        print(f"Steps where correct is longer: {pos}/{len(diffs)}")
        print(f"Steps where correct is shorter: {neg}/{len(diffs)}")

    # -----------------------------------------------------------------------
    # E) Length quantile analysis: what's happening at the distribution tails?
    # -----------------------------------------------------------------------
    print("\n### E: Length distribution by step (non-truncated only) ###\n")

    print(f"{'Step':>4} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8}")
    print("-" * 45)

    for step in sorted(episodes_by_step.keys()):
        eps = episodes_by_step[step]
        nt_eps = [e for e in eps if not e["truncated"]]
        if not nt_eps:
            continue
        lens = np.array([e["response_chars"] for e in nt_eps])
        print(f"{step:>4} {np.percentile(lens, 25):>8.0f} {np.percentile(lens, 50):>8.0f} "
              f"{np.percentile(lens, 75):>8.0f} {np.percentile(lens, 90):>8.0f} {np.percentile(lens, 95):>8.0f}")

    # -----------------------------------------------------------------------
    # F) The actual mechanism: is the model generating more NEAR-TRUNCATION
    #    responses? i.e., is the length distribution shifting rightward,
    #    pushing more mass past the 8192-token boundary?
    # -----------------------------------------------------------------------
    print("\n### F: Near-truncation analysis ###\n")

    metrics = load_metrics(LOG_DIR / "metrics.jsonl")

    # From metrics: total_ac_tokens / total_episodes = mean tokens per episode
    # If we assume max_tokens=8192, what fraction of the token budget is used?
    print(f"{'Step':>4} {'mean_tokens':>11} {'budget_used%':>12} {'trunc_rate%':>11}")
    print("-" * 40)
    for m in metrics:
        mean_toks = m["env/all/ac_tokens_per_turn"]
        budget_pct = 100 * mean_toks / 8192
        trunc_pct = 100 * (1 - m["env/all/format_eos"])
        print(f"{m['step']:>4} {mean_toks:>11.0f} {budget_pct:>11.1f}% {trunc_pct:>10.1f}%")

    # -----------------------------------------------------------------------
    # G) THE REAL STORY: reward signal breakdown
    # -----------------------------------------------------------------------
    print("\n\n### G: Understanding the actual gradient direction ###\n")
    print("MaxRL advantage with subgroups (alpha=0.5):")
    print("  - Split each group into top-50% and bottom-50% by reward")
    print("  - Compute advantages within each subgroup")
    print("  - This means truncated responses (-0.2) in the bottom subgroup")
    print("    have advantages centered around 0 (since they're all similar)")
    print("  - The actual gradient comes from the non-truncated responses")
    print("    where correct (reward=1.0) vs wrong (reward=0.0) differ")
    print()
    print("KEY INSIGHT: If correct non-truncated responses are SHORTER than")
    print("wrong non-truncated responses, PPO pushes toward shorter completions")
    print("among non-truncated. But if the model is also losing format compliance")
    print("(format_boxed dropping from 0.76 to 0.63), then more responses are")
    print("becoming unformatted/truncated. This could mean:")
    print("  1. The model learns 'shorter is better' for reasoning")
    print("  2. But shorter reasoning leads to worse format compliance")
    print("  3. Failed format → negative reward → more noise in gradient")
    print("  4. The MEAN length increases because truncated responses (which are")
    print("     all at max_tokens=8192) are a larger fraction of the total.")
    print()

    # Verify: is the increase in mean length ENTIRELY due to more truncation?
    print("Verification: mean length excluding truncated responses")
    print(f"{'Step':>4} {'mean_len_all':>12} {'mean_len_nontrunc':>17} {'trunc%':>7}")
    print("-" * 40)

    for step in sorted(episodes_by_step.keys()):
        eps = episodes_by_step[step]
        all_lens = np.mean([e["response_chars"] for e in eps])
        nt_eps = [e for e in eps if not e["truncated"]]
        if nt_eps:
            nt_lens = np.mean([e["response_chars"] for e in nt_eps])
        else:
            nt_lens = float("nan")
        trunc_pct = 100 * sum(1 for e in eps if e["truncated"]) / len(eps)
        print(f"{step:>4} {all_lens:>12.0f} {nt_lens:>17.0f} {trunc_pct:>6.1f}%")

    # Also check from metrics (full batch, not just logged subset)
    print("\nFrom metrics.jsonl (full 512-episode batches):")
    steps_arr = np.array([m["step"] for m in metrics])
    # Estimate non-truncated mean length:
    # total_ac_tokens = mean_tokens * total_episodes
    # truncated episodes use ~8192 tokens (at the limit)
    # non-truncated episodes use the remainder
    for m in metrics:
        total_eps = m["env/all/total_episodes"]
        eos_rate = m["env/all/format_eos"]
        n_nontrunc = total_eps * eos_rate
        n_trunc = total_eps * (1 - eos_rate)
        total_tokens = m["env/all/total_ac_tokens"]
        trunc_tokens = n_trunc * 8192  # each truncated episode maxes out the budget
        nontrunc_tokens = total_tokens - trunc_tokens
        if n_nontrunc > 0:
            mean_nontrunc = nontrunc_tokens / n_nontrunc
        else:
            mean_nontrunc = float("nan")
        # This is approximate since we don't know exact truncation length
        # But it gives the right direction

    # Simpler approach: use the metrics directly
    print(f"\n{'Step':>4} {'mean_all_tok':>12} {'est_nontrunc_tok':>16} {'trunc_rate':>10}")
    print("-" * 45)
    nontrunc_means = []
    for m in metrics:
        total_eps = m["env/all/total_episodes"]
        eos_rate = m["env/all/format_eos"]
        n_nontrunc = int(total_eps * eos_rate)
        n_trunc = total_eps - n_nontrunc
        total_tokens = m["env/all/total_ac_tokens"]
        trunc_tokens = n_trunc * 8192
        nontrunc_tokens = total_tokens - trunc_tokens
        if n_nontrunc > 0:
            mean_nt = nontrunc_tokens / n_nontrunc
        else:
            mean_nt = float("nan")
        nontrunc_means.append(mean_nt)
        trunc_pct = 100 * (1 - eos_rate)
        print(f"{m['step']:>4} {m['env/all/ac_tokens_per_turn']:>12.0f} {mean_nt:>16.0f} {trunc_pct:>9.1f}%")

    nontrunc_arr = np.array(nontrunc_means)
    corr_nt_step = np.corrcoef(steps_arr, nontrunc_arr)[0, 1]
    print(f"\nCorrelation (step, est_nontrunc_tokens): {corr_nt_step:.3f}")

    if abs(corr_nt_step) < 0.3:
        print(">>> Non-truncated length is NOT increasing!")
        print(">>> The increase in overall mean length is ENTIRELY due to more truncation")
    elif corr_nt_step > 0.3:
        print(">>> Non-truncated responses are also getting longer")
    else:
        print(">>> Non-truncated responses are getting SHORTER")

    print("\n\n" + "=" * 80)
    print("REVISED STORY")
    print("=" * 80)
    print("""
The original hypothesis is FALSE. Here's what's actually happening:

1. Correct responses are SHORTER than wrong ones (~1400 chars shorter).
   Within each step, the correlation between length and reward is strongly
   NEGATIVE (r=-0.49). PPO is being asked to push toward shorter responses.

2. Despite this, MEAN response length increases over training. This is
   because the TRUNCATION RATE is increasing (24% → 33%), and every
   truncated response maxes out at 8192 tokens. More truncation
   mechanically inflates the average.

3. The question is: WHY is truncation increasing if PPO penalizes it?
   Possible answers:
   a) PPO's gradient is too weak relative to the model's tendency to
      generate long thinking chains. The reward signal (-0.2 for truncation
      vs 0.0 for wrong) may not be strong enough.
   b) The entropy is rising (0.34 → 0.40), which means the model is
      becoming more random. More random sampling = more variance in length
      = more responses landing past the 8192 cutoff.
   c) Format compliance is dropping (76% → 66%), suggesting the model
      is LOSING learned behaviors, not gaining new ones. This looks more
      like training instability / reward hacking / KL divergence.

4. The ACTUAL degradation mechanism is likely:
   - PPO updates make the policy noisier (entropy rising)
   - Noisier policy → worse format compliance → more truncation
   - More truncation → lower correct rate
   - This is a "training instability" story, not a "length bias" story
""")


if __name__ == "__main__":
    main()
