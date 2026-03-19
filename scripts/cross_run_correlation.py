"""Cross-run correlation analysis: rung1 (no-think) vs rung2 (private-think).

Answers: do both runs fail on the same questions, or do training dynamics diverge?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Paths ──
BASE = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/thinking-experiment")
RUNG1 = BASE / "rung1-no-think" / "episodes" / "episodes.jsonl"
RUNG2 = BASE / "rung2-private-think" / "episodes" / "episodes.jsonl"
OUT_FIG = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/reports/figures/fig_cross_run_correlation.png")


def load_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def per_target_accuracy(episodes: list[dict]) -> dict[str, dict]:
    """Compute per-target accuracy, averaging accuracy.debater_a and accuracy.debater_b
    across all episodes for each target.

    Returns {target: {"acc": float, "n": int, "accs": list[float]}}
    """
    target_accs: dict[str, list[float]] = defaultdict(list)

    for ep in episodes:
        target = ep["target"]
        sigs = ep["signals"]
        # Each episode is from one debater's perspective. Use accuracy for that debater's role.
        role = ep["role"]
        acc_key = f"accuracy.{role}"
        if acc_key in sigs:
            target_accs[target].append(sigs[acc_key])

    result = {}
    for target, accs in target_accs.items():
        result[target] = {
            "acc": np.mean(accs),
            "n": len(accs),
            "accs": accs,
        }
    return result


def main():
    print("Loading episodes...")
    ep1 = load_episodes(RUNG1)
    ep2 = load_episodes(RUNG2)
    print(f"  rung1 (no-think):      {len(ep1):,} episodes")
    print(f"  rung2 (private-think): {len(ep2):,} episodes")

    # ── 1. Per-target accuracy ──
    acc1 = per_target_accuracy(ep1)
    acc2 = per_target_accuracy(ep2)

    targets1 = set(acc1.keys())
    targets2 = set(acc2.keys())
    shared = targets1 & targets2
    only1 = targets1 - targets2
    only2 = targets2 - targets1

    print(f"\n{'='*60}")
    print(f"QUESTION OVERLAP")
    print(f"{'='*60}")
    print(f"  Unique targets in rung1: {len(targets1)}")
    print(f"  Unique targets in rung2: {len(targets2)}")
    print(f"  Shared targets:          {len(shared)}")
    print(f"  Only in rung1:           {len(only1)}")
    print(f"  Only in rung2:           {len(only2)}")

    # ── 2. Build paired arrays ──
    shared_sorted = sorted(shared)
    x = np.array([acc1[t]["acc"] for t in shared_sorted])
    y = np.array([acc2[t]["acc"] for t in shared_sorted])
    n1 = np.array([acc1[t]["n"] for t in shared_sorted])
    n2 = np.array([acc2[t]["n"] for t in shared_sorted])

    print(f"\n  Episodes per target (rung1): median={np.median(n1):.0f}, min={n1.min()}, max={n1.max()}")
    print(f"  Episodes per target (rung2): median={np.median(n2):.0f}, min={n2.min()}, max={n2.max()}")

    # ── 3. Correlations ──
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    r_squared = pearson_r ** 2

    print(f"\n{'='*60}")
    print(f"CORRELATION (per-target accuracy, n={len(shared)} questions)")
    print(f"{'='*60}")
    print(f"  Pearson r:  {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman r: {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"  R²:         {r_squared:.4f}")
    print(f"  → {r_squared*100:.1f}% of variance in per-target accuracy is shared")
    print(f"  → {(1-r_squared)*100:.1f}% is training-path-dependent or noise")

    # ── 4. Scatter plot ──
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use colorblind-safe colors (Okabe-Ito)
    # Size points by sqrt of total episodes
    sizes = np.sqrt(n1 + n2) * 5

    ax.scatter(x, y, s=sizes, alpha=0.5, c="#0072B2", edgecolors="#0072B2", linewidths=0.5)

    # Diagonal
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="y = x")

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "-", color="#D55E00", linewidth=1.5,
            label=f"OLS: y = {slope:.2f}x + {intercept:.2f}")

    ax.set_xlabel("Rung 1 (no-think) per-target accuracy", fontsize=12)
    ax.set_ylabel("Rung 2 (private-think) per-target accuracy", fontsize=12)
    ax.set_title(
        f"Cross-run accuracy correlation (n={len(shared)} questions)\n"
        f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}, R²={r_squared:.3f}",
        fontsize=11,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=600)
    print(f"\n  Saved scatter plot → {OUT_FIG}")

    # ── 5. Where thinking HELPS (rung2 > rung1 by >0.3) ──
    print(f"\n{'='*60}")
    print(f"QUESTIONS WHERE THINKING HELPS (rung2 - rung1 > 0.3)")
    print(f"{'='*60}")

    gaps = []
    for t in shared_sorted:
        gap = acc2[t]["acc"] - acc1[t]["acc"]
        gaps.append((gap, t))

    helps = sorted([g for g in gaps if g[0] > 0.3], key=lambda x: -x[0])
    print(f"  Found {len(helps)} questions")
    for i, (gap, t) in enumerate(helps[:10]):
        print(f"  {i+1:2d}. gap={gap:+.3f}  r1={acc1[t]['acc']:.3f} (n={acc1[t]['n']})  "
              f"r2={acc2[t]['acc']:.3f} (n={acc2[t]['n']})")
        # Truncate target for display
        tdisp = t[:120] + "..." if len(t) > 120 else t
        print(f"      target: {tdisp}")

    # ── 6. Where thinking HURTS (rung1 > rung2 by >0.3) ──
    print(f"\n{'='*60}")
    print(f"QUESTIONS WHERE THINKING HURTS (rung1 - rung2 > 0.3)")
    print(f"{'='*60}")

    hurts = sorted([(-g[0], g[1]) for g in gaps if g[0] < -0.3], key=lambda x: -x[0])
    print(f"  Found {len(hurts)} questions")
    for i, (gap, t) in enumerate(hurts[:10]):
        print(f"  {i+1:2d}. gap={gap:+.3f}  r1={acc1[t]['acc']:.3f} (n={acc1[t]['n']})  "
              f"r2={acc2[t]['acc']:.3f} (n={acc2[t]['n']})")
        tdisp = t[:120] + "..." if len(t) > 120 else t
        print(f"      target: {tdisp}")

    # ── 7. Distribution summary ──
    print(f"\n{'='*60}")
    print(f"ACCURACY DISTRIBUTIONS")
    print(f"{'='*60}")
    print(f"  Rung1: mean={x.mean():.3f}, median={np.median(x):.3f}, std={x.std():.3f}")
    print(f"  Rung2: mean={y.mean():.3f}, median={np.median(y):.3f}, std={y.std():.3f}")
    print(f"  Mean gap (rung2 - rung1): {(y - x).mean():.3f} ± {(y - x).std():.3f}")

    # Bin analysis
    both_fail = np.sum((x < 0.3) & (y < 0.3))
    both_pass = np.sum((x > 0.7) & (y > 0.7))
    r1_only = np.sum((x > 0.7) & (y < 0.3))
    r2_only = np.sum((x < 0.3) & (y > 0.7))
    print(f"\n  Quadrant analysis (thresholds: fail<0.3, pass>0.7):")
    print(f"    Both fail:       {both_fail:3d} ({both_fail/len(shared)*100:.1f}%)")
    print(f"    Both pass:       {both_pass:3d} ({both_pass/len(shared)*100:.1f}%)")
    print(f"    Rung1 pass only: {r1_only:3d} ({r1_only/len(shared)*100:.1f}%)")
    print(f"    Rung2 pass only: {r2_only:3d} ({r2_only/len(shared)*100:.1f}%)")

    # ── 8. Save targets where thinking helps most for transcript analysis ──
    # Save top 5 targets where thinking helps most
    help_targets = [t for _, t in helps[:5]]
    with open(BASE / "thinking_helps_targets.json", "w") as f:
        json.dump(help_targets, f, indent=2)
    print(f"\n  Saved top-5 thinking-helps targets → {BASE / 'thinking_helps_targets.json'}")


if __name__ == "__main__":
    main()
