"""Generate training dynamics figures from debate experiment metrics."""

import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

rung1 = load_jsonl(BASE / "logs/thinking-experiment/rung1-no-think/metrics.jsonl")
rung2 = load_jsonl(BASE / "logs/thinking-experiment/rung2-private-think/metrics.jsonl")

# ── Helpers ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = plt.cm.tab10.colors  # tableau-colorblind10 compatible
C_R1_A, C_R1_B = "#1f77b4", "#2171b5"  # blue family for rung1 (darker KL axis)
C_R2_A, C_R2_B = "#e6550d", "#a63603"  # orange family for rung2 (darker KL axis)
C_R1, C_R2 = "#1f77b4", "#e6550d"

def get(rows, key, prefix="env/all/"):
    """Extract a metric series. Try env/all/ first, fall back to test/env/all/."""
    vals = []
    for r in rows:
        full = prefix + key
        if full in r:
            v = r[full]
            vals.append(v if not (isinstance(v, float) and math.isnan(v)) else None)
        else:
            # try without prefix
            if key in r:
                v = r[key]
                vals.append(v if not (isinstance(v, float) and math.isnan(v)) else None)
            else:
                vals.append(None)
    return vals

def steps(rows):
    return [r["step"] for r in rows]

def savefig(fig, name):
    fig.savefig(OUT / name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

# ── Figure 1: Accuracy ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, ca, cb in [
    (ax1, rung1, "Rung 1: No Think", C_R1_A, C_R1_B),
    (ax2, rung2, "Rung 2: Private Think", C_R2_A, C_R2_B),
]:
    s = steps(data)
    acc_a = get(data, "accuracy.debater_a")
    acc_b = get(data, "accuracy.debater_b")
    ax.plot(s, acc_a, "o-", color=ca, label="Debater A accuracy", markersize=4)
    ax.plot(s, acc_b, "s-", color=cb, label="Debater B accuracy", markersize=4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("Accuracy", fontsize=10)

# Annotate collapse in rung1
ax1.annotate("Accuracy collapse\n(both → ~0.25)",
             xy=(10, 0.25), xytext=(6, 0.08),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
             color="red")

fig.suptitle("Debater Accuracy Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_accuracy.png")

# ── Figure 2: Disagreement ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, c in [
    (ax1, rung1, "Rung 1: No Think", C_R1),
    (ax2, rung2, "Rung 2: Private Think", C_R2),
]:
    s = steps(data)
    dis = get(data, "disagreement")
    ax.plot(s, dis, "o-", color=c, markersize=4, label="Disagreement rate")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("Disagreement Rate", fontsize=10)

# Annotate the collapse in rung1 (step 9 where it drops to ~0.047)
ax1.annotate("Collapse → 0.047",
             xy=(9, 0.047), xytext=(6, 0.55),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
             color="red")

fig.suptitle("Disagreement Rate Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_disagreement.png")

# ── Figure 3: Draw Rate ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, c in [
    (ax1, rung1, "Rung 1: No Think", C_R1),
    (ax2, rung2, "Rung 2: Private Think", C_R2),
]:
    s = steps(data)
    dr = get(data, "draw_rate")
    ax.plot(s, dr, "o-", color=c, markersize=4, label="Draw rate")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("Draw Rate", fontsize=10)

# Annotate peaks
# rung1 peak: step 1 at 0.605
r1_dr = get(rung1, "draw_rate")
r1_max_idx = max(range(len(r1_dr)), key=lambda i: r1_dr[i] if r1_dr[i] is not None else -1)
r1_max_val = r1_dr[r1_max_idx]
ax1.annotate(f"Peak: {r1_max_val:.0%}",
             xy=(steps(rung1)[r1_max_idx], r1_max_val),
             xytext=(steps(rung1)[r1_max_idx] + 2, r1_max_val + 0.1),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

# rung2 peak
r2_dr = get(rung2, "draw_rate")
r2_max_idx = max(range(len(r2_dr)), key=lambda i: r2_dr[i] if r2_dr[i] is not None else -1)
r2_max_val = r2_dr[r2_max_idx]
ax2.annotate(f"Peak: {r2_max_val:.0%}",
             xy=(steps(rung2)[r2_max_idx], r2_max_val),
             xytext=(steps(rung2)[r2_max_idx] + 1, r2_max_val + 0.12),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

fig.suptitle("Draw Rate Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_draw_rate.png")

# ── Figure 4: Entropy & KL ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# Use same dark colors for lines and axis labels
_ek_colors = [
    (ax1, rung1, "Rung 1: No Think", "#1f77b4", "#1f77b4"),   # both dark blue
    (ax2, rung2, "Rung 2: Private Think", "#e6550d", "#e6550d"),  # both dark orange
]
for ax, data, title, c_ent, c_kl in _ek_colors:
    s = steps(data)
    entropy = [r.get("optim/entropy") for r in data]
    kl = [r.get("optim/kl_sample_train_v2") for r in data]

    ax.plot(s, entropy, "o-", color=c_ent, markersize=4, label="Entropy")
    ax.set_ylabel("Entropy", fontsize=10, color=c_ent)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelcolor=c_ent)
    ax.grid(True, alpha=0.3)

    ax2r = ax.twinx()
    ax2r.plot(s, kl, "s--", color=c_kl, markersize=4, label="KL divergence")
    ax2r.set_ylabel("KL divergence", fontsize=10, color=c_kl)
    ax2r.tick_params(axis="y", labelcolor=c_kl)

    # Legend below x-axis, outside plot area
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

fig.suptitle("Entropy & KL Divergence Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_entropy_kl.png")

# ── Figure 5: Tokens per Turn ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, ca, cb in [
    (ax1, rung1, "Rung 1: No Think", C_R1_A, C_R1_B),
    (ax2, rung2, "Rung 2: Private Think", C_R2_A, C_R2_B),
]:
    s = steps(data)
    ac_tok = get(data, "ac_tokens_per_turn")
    ob_tok = get(data, "ob_tokens_per_turn")
    ax.plot(s, ac_tok, "o-", color=ca, markersize=4, label="Action tokens/turn")
    ax.plot(s, ob_tok, "s-", color=cb, markersize=4, label="Observation tokens/turn")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Context limit line: 32k / (4 turns * 2 sides) ≈ 4000 tok/turn
    ax.axhline(y=4000, color="red", linestyle="--", alpha=0.7, linewidth=1.2)
    # Place label at midpoint of x-range to avoid clipping at edges
    x_mid = s[len(s) // 2]
    ax.text(x_mid, 4100, "~Context budget", fontsize=8, color="red", ha="center", va="bottom")

ax1.set_ylabel("Tokens per Turn", fontsize=10)

fig.suptitle("Tokens per Turn Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_tokens_growth.png")

# ── Figure 6: Reward ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, c in [
    (ax1, rung1, "Rung 1: No Think", C_R1),
    (ax2, rung2, "Rung 2: Private Think", C_R2),
]:
    s = steps(data)
    reward = get(data, "reward/total")
    ax.plot(s, reward, "o-", color=c, markersize=4, label="Reward (total)")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("Reward", fontsize=10)

# Annotate flatline in rung1
ax1.annotate("Reward → 0\n(no signal)",
             xy=(9, 0.0), xytext=(6, -0.025),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
             color="red")

fig.suptitle("Reward Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_reward.png")

# ── Figure 7: Win Rate by Seat ──────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, ca, cb in [
    (ax1, rung1, "Rung 1: No Think", C_R1_A, C_R1_B),
    (ax2, rung2, "Rung 2: Private Think", C_R2_A, C_R2_B),
]:
    s = steps(data)
    win_a = get(data, "win_rate.debater_a")
    win_b = get(data, "win_rate.debater_b")
    dr = get(data, "draw_rate")

    ax.plot(s, win_a, "o-", color=ca, markersize=4, label="Win rate A")
    ax.plot(s, win_b, "s-", color=cb, markersize=4, label="Win rate B")
    ax.plot(s, dr, "^--", color="gray", markersize=4, alpha=0.6, label="Draw rate")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Shade the B dominance region
    ax.fill_between(s, win_a, win_b, alpha=0.15, color=cb, label="_nolegend_")

ax1.set_ylabel("Rate", fontsize=10)

# Annotate B dominance
ax1.annotate("B dominates",
             xy=(12, 0.55), fontsize=10, color="navy",
             ha="center", weight="bold")

fig.suptitle("Win Rate by Seat Over Training", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_win_rate_seats.png")

# ── Figure 8: Seat Bias Trend ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for data, label, c, marker in [
    (rung1, "Rung 1: No Think", C_R1, "o"),
    (rung2, "Rung 2: Private Think", C_R2, "s"),
]:
    s = steps(data)
    win_a = get(data, "win_rate.debater_a")
    win_b = get(data, "win_rate.debater_b")
    bias = [b - a if (a is not None and b is not None) else None for a, b in zip(win_a, win_b)]
    ax.plot(s, bias, f"{marker}-", color=c, markersize=5, label=label)

ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
ax.set_xlabel("Training Step", fontsize=10)
ax.set_ylabel("Seat Bias (win_B - win_A)", fontsize=10)
ax.set_title("Seat Bias Over Training", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax.annotate("Persistent B advantage\nacross both conditions",
            xy=(10, 0.35), fontsize=10, color="navy",
            ha="center", style="italic")

fig.tight_layout()
savefig(fig, "fig_seat_bias_trend.png")

print("\nAll figures generated.")
