"""Generate training dynamics figures for the protocol experiment.

Reads metrics_all_runs.csv and produces:
  - fig_protocol_accuracy.png
  - fig_protocol_judge_quality.png
  - fig_protocol_reward.png
  - fig_protocol_kl_entropy.png
  - fig_protocol_truth_win.png
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from style import apply_style, COLORS, annotate_insight, save

apply_style()

# Protocol colors
PROTO_COLORS = {
    "seq-v1": COLORS["blue"],
    "seq-v2": COLORS["orange"],
    "hybrid": COLORS["green"],
    "simultaneous": COLORS["red"],
}
PROTO_LABELS = {
    "seq-v1": "Sequential v1",
    "seq-v2": "Sequential v2",
    "hybrid": "Hybrid",
    "simultaneous": "Simultaneous",
}
RUNS = ["seq-v1", "seq-v2", "hybrid", "simultaneous"]

df = pd.read_csv("reports/protocol-experiment/metrics_all_runs.csv")


def plot_metric(ax, col, ylabel, title, annotate_peaks=False, annotate_events=True):
    """Plot a single metric across all runs."""
    for run in RUNS:
        rd = df[df["run"] == run].sort_values("step")
        vals = rd[col].values
        steps = rd["step"].values
        mask = ~np.isnan(vals)
        ax.plot(
            steps[mask],
            vals[mask],
            marker="o",
            markersize=5,
            linewidth=2,
            color=PROTO_COLORS[run],
            label=PROTO_LABELS[run],
        )
        if annotate_peaks and len(vals[mask]) > 0:
            peak_idx = np.nanargmax(vals)
            ax.plot(
                steps[peak_idx],
                vals[peak_idx],
                marker="*",
                markersize=12,
                color=PROTO_COLORS[run],
                zorder=5,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")


# --- Figure 1: Accuracy ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for run in RUNS:
    rd = df[df["run"] == run].sort_values("step")
    a_vals = rd["env/all/accuracy.debater_a"].values
    b_vals = rd["env/all/accuracy.debater_b"].values
    mean_acc = (a_vals + b_vals) / 2
    steps = rd["step"].values
    mask = ~np.isnan(mean_acc)
    axes[0].plot(
        steps[mask],
        mean_acc[mask],
        marker="o",
        markersize=5,
        linewidth=2,
        color=PROTO_COLORS[run],
        label=PROTO_LABELS[run],
    )

axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Mean Debater Accuracy")
axes[0].set_title("Debater Accuracy (mean of A & B)")
axes[0].legend(loc="best")
axes[0].axhline(y=0.5, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.7)

# Accuracy delta (A - B) to show seat bias
for run in RUNS:
    rd = df[df["run"] == run].sort_values("step")
    delta = rd["env/all/accuracy.debater_a"].values - rd["env/all/accuracy.debater_b"].values
    steps = rd["step"].values
    mask = ~np.isnan(delta)
    axes[1].plot(
        steps[mask],
        delta[mask],
        marker="o",
        markersize=5,
        linewidth=2,
        color=PROTO_COLORS[run],
        label=PROTO_LABELS[run],
    )

axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Accuracy(A) - Accuracy(B)")
axes[1].set_title("Seat Accuracy Bias (A - B)")
axes[1].axhline(y=0, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.7)
axes[1].legend(loc="best")

fig.tight_layout()
save(fig, "fig_protocol_accuracy")
plt.close(fig)


# --- Figure 2: Judge Quality ---
fig, ax = plt.subplots(figsize=(8, 5))
plot_metric(
    ax,
    "env/all/judge_quality",
    "Judge Quality",
    "Judge Quality Over Training",
    annotate_peaks=True,
)
ax.axhline(y=0.5, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.7)
# Annotate simultaneous peak at step 6
sim = df[df["run"] == "simultaneous"].sort_values("step")
peak_step = sim.loc[sim["env/all/judge_quality"].idxmax(), "step"]
peak_val = sim["env/all/judge_quality"].max()
annotate_insight(
    ax,
    f"Peak {peak_val:.3f}",
    (peak_step, peak_val),
    (peak_step - 1.5, peak_val + 0.06),
    color=PROTO_COLORS["simultaneous"],
)
fig.tight_layout()
save(fig, "fig_protocol_judge_quality")
plt.close(fig)


# --- Figure 3: Reward ---
fig, ax = plt.subplots(figsize=(8, 5))
plot_metric(ax, "env/all/reward/total", "Reward", "Training Reward Over Steps")
ax.axhline(y=0, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.7)
# Annotate simultaneous steep decline
annotate_insight(
    ax,
    "Steep decline",
    (5, -0.11),
    (3, -0.12),
    color=PROTO_COLORS["simultaneous"],
)
fig.tight_layout()
save(fig, "fig_protocol_reward")
plt.close(fig)


# --- Figure 4: KL & Entropy (separate panels) ---
fig, (ax_kl, ax_ent) = plt.subplots(1, 2, figsize=(14, 5))

for run in RUNS:
    rd = df[df["run"] == run].sort_values("step")
    steps = rd["step"].values

    kl = rd["optim/kl_sample_train_v1"].values
    mask_kl = ~np.isnan(kl)
    ax_kl.plot(
        steps[mask_kl],
        kl[mask_kl],
        marker="o",
        markersize=5,
        linewidth=2,
        color=PROTO_COLORS[run],
        label=PROTO_LABELS[run],
    )

    ent = rd["optim/entropy"].values
    mask_ent = ~np.isnan(ent)
    ax_ent.plot(
        steps[mask_ent],
        ent[mask_ent],
        marker="o",
        markersize=5,
        linewidth=2,
        color=PROTO_COLORS[run],
        label=PROTO_LABELS[run],
    )

ax_kl.set_xlabel("Training Step")
ax_kl.set_ylabel("KL Divergence (sample train v1)")
ax_kl.set_title("KL Divergence Over Training")
ax_kl.legend(loc="best")

ax_ent.set_xlabel("Training Step")
ax_ent.set_ylabel("Entropy")
ax_ent.set_title("Entropy Over Training")
ax_ent.legend(loc="best")

# Hybrid KL spike at step 2
hybrid_kl = df[df["run"] == "hybrid"].sort_values("step")["optim/kl_sample_train_v1"]
if hybrid_kl.max() > 0.01:
    annotate_insight(
        ax_kl,
        f"KL spike {hybrid_kl.max():.4f}",
        (2, hybrid_kl.max()),
        (4, hybrid_kl.max() + 0.002),
        color=PROTO_COLORS["hybrid"],
    )

fig.tight_layout()
save(fig, "fig_protocol_kl_entropy")
plt.close(fig)


# --- Figure 5: Truth wins if disagreement ---
# Scatter + Wilson CI to honestly represent small-N uncertainty.
# The truth_win_if_disagreement metric conditions on disagreement AND
# both having answers AND exactly one correct — often single-digit N.

def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return np.nan, np.nan, np.nan
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center, max(0, center - spread), min(1, center + spread)


def infer_n(p):
    """Infer smallest integer N such that p = k/N for some integer k."""
    if np.isnan(p):
        return 0
    for n in range(1, 500):
        k = round(p * n)
        if abs(k / n - p) < 1e-9:
            return n
    return 0


fig, ax = plt.subplots(figsize=(9, 5.5))

# Slight x-jitter to prevent overlap between runs at same step
jitter = {"seq-v1": -0.15, "seq-v2": -0.05, "hybrid": 0.05, "simultaneous": 0.15}

for run in RUNS:
    rd = df[df["run"] == run].sort_values("step")
    col = "env/all/truth_win_if_disagreement"
    vals = rd[col].values.astype(float)
    steps = rd["step"].values

    valid = ~np.isnan(vals)
    if not np.any(valid):
        continue

    v_steps = steps[valid].astype(float) + jitter[run]
    v_vals = vals[valid]

    # Compute exact N from the proportion values
    exact_n = np.array([infer_n(p) for p in v_vals])

    ci_lo = []
    ci_hi = []
    for p, n in zip(v_vals, exact_n):
        _, lo, hi = wilson_ci(p, n)
        ci_lo.append(lo)
        ci_hi.append(hi)

    yerr_lo = np.maximum(0, v_vals - np.array(ci_lo))
    yerr_hi = np.maximum(0, np.array(ci_hi) - v_vals)
    ax.errorbar(
        v_steps, v_vals,
        yerr=[yerr_lo, yerr_hi],
        fmt="o", markersize=7, capsize=4, linewidth=1.5,
        color=PROTO_COLORS[run], label=PROTO_LABELS[run],
        alpha=0.85, capthick=1.2, zorder=3,
    )
    # N annotations
    for s, v, n in zip(v_steps, v_vals, exact_n):
        ax.annotate(
            f"n={n}", (s, v), textcoords="offset points",
            xytext=(8, -10), fontsize=6.5, color=PROTO_COLORS[run], alpha=0.75,
        )

ax.set_xlabel("Training Step")
ax.set_ylabel("P(Truth Wins | Disagreement)")
ax.set_title("Truth Win Rate When Debaters Disagree")
ax.axhline(y=0.5, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_ylim(-0.1, 1.15)
ax.legend(loc="upper right", fontsize=9)

# Add a note about small N
ax.text(
    0.02, 0.02,
    "Caution: N is very small (1-15 episodes per point).\n"
    "Error bars show 95% Wilson score CI.",
    transform=ax.transAxes, fontsize=7.5, color=COLORS["light"],
    va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.9),
)

fig.tight_layout()
save(fig, "fig_protocol_truth_win")
plt.close(fig)

print("All 5 protocol training figures generated.")
