"""Parametric throughput plots for RLVR and Debate training.

Run: uv run python scripts/plot_throughput.py
Generates: scripts/throughput_plots.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Derived invariants from smoke tests
# ---------------------------------------------------------------------------

# RLVR invariants (from Qwen3-8B smoke tests)
RLVR_CONFIGS = {
    "Qwen3-8B think": {"t_sample_baseline": 317, "t_train_per_64": 9.5, "T_avg": 6418, "color": "#e74c3c"},
    "Qwen3-8B no-think": {"t_sample_baseline": 147, "t_train_per_64": 3.6, "T_avg": 1215, "color": "#2ecc71"},
    "GPT-OSS-20B medium": {"t_sample_baseline": 188, "t_train_per_64": 6.4, "T_avg": 3833, "color": "#3498db"},
}

# Debate invariants (estimated from code structure + bottleneck findings)
DEBATE_T_S = 10  # seconds per debater sampling turn (mid-size model)
DEBATE_T_J = 10  # seconds per judge call
DEBATE_T_TRAIN_PER_DATUM = 0.02  # seconds per training datum (clock cycle amortized)
DEBATE_T_CKPT = 3  # seconds per checkpoint save

# Service capacity
C_DEFAULT = 256  # sampling_max_connections


def rlvr_t_batch(BG, t_sample_baseline, t_train_per_64, batched=False):
    """RLVR batch wall time.

    t_sample_baseline: measured at BG=64 with unbatched rollout.
    batched: if True, assume num_samples=G reduces sampling by factor.
    """
    # Sampling: dominated by slowest completion (≈ constant in BG if all parallel)
    # But service throughput caps parallelism
    t_sample = t_sample_baseline  # approximately constant (all parallel, wait for slowest)
    if batched:
        # Estimated 2-3x speedup from batched decode (conservative: 2x)
        t_sample = t_sample_baseline / 2.0
    # Training: linear in BG
    t_train = t_train_per_64 * (BG / 64)
    return t_sample + t_train


def debate_t_batch(B, G, R, protocol, C=C_DEFAULT, t_s=DEBATE_T_S, t_j=DEBATE_T_J):
    """Debate batch wall time (self-play)."""
    if protocol == "SEQUENTIAL":
        t_episode = 2 * R * t_s + t_j
        calls_per_ep = 2 * R + 1
    elif protocol == "SIMULTANEOUS":
        t_episode = R * t_s + t_j
        calls_per_ep = R + 1
    else:
        raise ValueError(protocol)

    # Two regimes: critical path vs throughput limit
    t_throughput = B * G * calls_per_ep * t_s / C
    t_rollout = max(t_episode, t_throughput)

    # Training: 2 trajectories per game (self-play), R transitions per trajectory
    n_datums = B * G * 2 * R
    t_train = n_datums * DEBATE_T_TRAIN_PER_DATUM

    return t_rollout + t_train + DEBATE_T_CKPT


def hours(seconds, N_batches):
    return seconds * N_batches / 3600


# ---------------------------------------------------------------------------
# Figure layout: 2x3 grid
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Parametric Throughput Analysis: RLVR & Debate", fontsize=14, fontweight="bold")

# ---------------------------------------------------------------------------
# Plot 1: RLVR — T_batch vs B·G (unbatched vs batched)
# ---------------------------------------------------------------------------

ax = axes[0, 0]
BG_range = np.arange(16, 513, 16)

for name, cfg in RLVR_CONFIGS.items():
    t_unbatched = [rlvr_t_batch(bg, cfg["t_sample_baseline"], cfg["t_train_per_64"], batched=False) for bg in BG_range]
    t_batched = [rlvr_t_batch(bg, cfg["t_sample_baseline"], cfg["t_train_per_64"], batched=True) for bg in BG_range]
    ax.plot(BG_range, t_unbatched, "--", color=cfg["color"], alpha=0.5, label=f"{name} (unbatched)")
    ax.plot(BG_range, t_batched, "-", color=cfg["color"], linewidth=2, label=f"{name} (num_samples=G)")

ax.set_xlabel("B·G (total samples per batch)")
ax.set_ylabel("T_batch (seconds)")
ax.set_title("RLVR: Batch Wall Time")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2: RLVR — Total training time for N=100 batches
# ---------------------------------------------------------------------------

ax = axes[0, 1]
N = 100

for name, cfg in RLVR_CONFIGS.items():
    t_batched = [hours(rlvr_t_batch(bg, cfg["t_sample_baseline"], cfg["t_train_per_64"], batched=True), N) for bg in BG_range]
    ax.plot(BG_range, t_batched, "-", color=cfg["color"], linewidth=2, label=name)

ax.set_xlabel("B·G (total samples per batch)")
ax.set_ylabel(f"Total time ({N} batches, hours)")
ax.set_title(f"RLVR: {N}-Batch Run Time (batched)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=4, color="gray", linestyle=":", alpha=0.5, label="4h target")

# ---------------------------------------------------------------------------
# Plot 3: RLVR — Gradient signal per hour (sqrt(BG) / T_batch)
# ---------------------------------------------------------------------------

ax = axes[0, 2]

for name, cfg in RLVR_CONFIGS.items():
    efficiency = [np.sqrt(bg) / rlvr_t_batch(bg, cfg["t_sample_baseline"], cfg["t_train_per_64"], batched=True) * 3600
                  for bg in BG_range]
    ax.plot(BG_range, efficiency, "-", color=cfg["color"], linewidth=2, label=name)

ax.set_xlabel("B·G (total samples per batch)")
ax.set_ylabel("Gradient quality per hour (√(B·G) / T_batch × 3600)")
ax.set_title("RLVR: Learning Efficiency")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 4: Debate — T_batch heatmap (B vs G, SEQUENTIAL R=2)
# ---------------------------------------------------------------------------

ax = axes[1, 0]
B_range = np.array([4, 8, 16, 32, 64, 128])
G_range = np.array([2, 4, 8, 16])

heatmap = np.zeros((len(G_range), len(B_range)))
for i, G in enumerate(G_range):
    for j, B in enumerate(B_range):
        heatmap[i, j] = debate_t_batch(B, G, R=2, protocol="SEQUENTIAL")

im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", origin="lower")
ax.set_xticks(range(len(B_range)))
ax.set_xticklabels(B_range)
ax.set_yticks(range(len(G_range)))
ax.set_yticklabels(G_range)
ax.set_xlabel("B (batch_size)")
ax.set_ylabel("G (group_size)")
ax.set_title("Debate SEQUENTIAL R=2: T_batch (s)")
for i in range(len(G_range)):
    for j in range(len(B_range)):
        ax.text(j, i, f"{heatmap[i,j]:.0f}s", ha="center", va="center", fontsize=7,
                color="white" if heatmap[i,j] > heatmap.max()*0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)

# ---------------------------------------------------------------------------
# Plot 5: Debate — SEQUENTIAL vs SIMULTANEOUS vs rounds
# ---------------------------------------------------------------------------

ax = axes[1, 1]
B_sweep = np.arange(4, 129, 4)

for R in [1, 2, 3]:
    for protocol, ls in [("SEQUENTIAL", "-"), ("SIMULTANEOUS", "--")]:
        t = [debate_t_batch(B, G=4, R=R, protocol=protocol) for B in B_sweep]
        label = f"{protocol[:3]} R={R}"
        color = {1: "#2ecc71", 2: "#3498db", 3: "#e74c3c"}[R]
        ax.plot(B_sweep, t, ls, color=color, linewidth=2, label=label)

ax.set_xlabel("B (batch_size, G=4 fixed)")
ax.set_ylabel("T_batch (seconds)")
ax.set_title("Debate: Protocol × Rounds (G=4)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 6: Debate — Regime crossover (critical path vs throughput)
# ---------------------------------------------------------------------------

ax = axes[1, 2]
BG_range_d = np.arange(8, 1025, 8)
R = 2

for protocol, color, ls in [("SEQUENTIAL", "#e74c3c", "-"), ("SIMULTANEOUS", "#2ecc71", "--")]:
    if protocol == "SEQUENTIAL":
        t_ep = 2 * R * DEBATE_T_S + DEBATE_T_J
        calls = 2 * R + 1
    else:
        t_ep = R * DEBATE_T_S + DEBATE_T_J
        calls = R + 1

    t_critical = [t_ep] * len(BG_range_d)
    t_throughput = [bg * calls * DEBATE_T_S / C_DEFAULT for bg in BG_range_d]
    t_total = [max(tc, tt) for tc, tt in zip(t_critical, t_throughput)]

    ax.plot(BG_range_d, t_critical, ":", color=color, alpha=0.5, label=f"{protocol[:3]} critical path")
    ax.plot(BG_range_d, t_throughput, ":", color=color, alpha=0.5)
    ax.plot(BG_range_d, t_total, ls, color=color, linewidth=2, label=f"{protocol[:3]} T_rollout")

    # Mark crossover
    crossover = t_ep * C_DEFAULT / (calls * DEBATE_T_S)
    ax.axvline(x=crossover, color=color, linestyle="-.", alpha=0.3)
    ax.annotate(f"B·G={crossover:.0f}", xy=(crossover, t_ep), fontsize=7,
                xytext=(crossover + 50, t_ep + 20), arrowprops=dict(arrowstyle="->", color=color))

ax.set_xlabel("B·G (total games per batch)")
ax.set_ylabel("T_rollout (seconds)")
ax.set_title("Debate R=2: Regime Crossover (C=256)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.tight_layout()
plt.savefig("scripts/throughput_plots.png", dpi=150, bbox_inches="tight")
print("Saved to scripts/throughput_plots.png")
plt.close()
