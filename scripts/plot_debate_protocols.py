"""Detailed debate protocol comparison plots.

Three protocols test different things:
- SEQUENTIAL: agents take turns, full information at each step
- SIMULTANEOUS: agents act in parallel, no peeking at opponent's current move
- (Frozen-opponent variants of each)

Run: uv run python scripts/plot_debate_protocols.py
Generates: scripts/debate_protocol_plots.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

T_S_RANGE = [5, 10, 15, 20]  # seconds per debater turn (varies with model size)
T_J = 10  # judge call (relatively stable)
T_TRAIN_PER_DATUM = 0.02
T_CKPT = 3
C = 256  # service concurrency


def t_episode(R, protocol, frozen, t_s, t_j=T_J):
    """Critical path for one debate episode."""
    if protocol == "SEQ":
        if frozen:
            # Trained agent: R turns. Opponent: R turns (sequential, different model)
            t_opp = t_s * 0.7  # opponent often smaller/faster
            return R * t_s + R * t_opp + t_j
        else:
            return 2 * R * t_s + t_j  # self-play: 2R turns, same model
    elif protocol == "SIM":
        if frozen:
            # Each round: max(trained, opponent) ≈ t_s
            return R * t_s + t_j
        else:
            return R * t_s + t_j  # both sample in parallel per round
    raise ValueError(protocol)


def calls_per_episode(R, protocol, frozen):
    """Total Tinker API calls per episode."""
    if protocol == "SEQ":
        if frozen:
            return R + R + 1  # R trained + R opponent + 1 judge
        return 2 * R + 1  # self-play: 2R + judge
    elif protocol == "SIM":
        if frozen:
            return R + R + 1  # still R+R calls, just overlapped in time
        return 2 * R + 1  # both fire per round + judge
    raise ValueError(protocol)


def t_batch(B, G, R, protocol, frozen, t_s, C=C):
    """Total batch wall time."""
    t_ep = t_episode(R, protocol, frozen, t_s)
    n_calls = calls_per_episode(R, protocol, frozen)
    t_throughput = B * G * n_calls * t_s / C
    t_rollout = max(t_ep, t_throughput)

    roles = 1 if frozen else 2
    n_datums = B * G * roles * R
    t_train = n_datums * T_TRAIN_PER_DATUM
    return t_rollout + t_train + T_CKPT


def total_hours(B, G, R, protocol, frozen, t_s, N):
    return t_batch(B, G, R, protocol, frozen, t_s) * N / 3600


# ---------------------------------------------------------------------------
# Figure: 2x3 grid
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Debate Training: Protocol × Scaling Analysis", fontsize=14, fontweight="bold")

# ---------------------------------------------------------------------------
# Plot 1: Protocol comparison — T_batch vs B (fixed G=4, R=2)
# ---------------------------------------------------------------------------

ax = axes[0, 0]
B_range = np.arange(4, 129, 4)
G_fixed = 4
R_fixed = 2
t_s_fixed = 10

configs = [
    ("SEQ self-play", "SEQ", False, "#e74c3c", "-"),
    ("SEQ frozen-opp", "SEQ", True, "#e74c3c", "--"),
    ("SIM self-play", "SIM", False, "#2ecc71", "-"),
    ("SIM frozen-opp", "SIM", True, "#2ecc71", "--"),
]

for name, proto, frozen, color, ls in configs:
    t = [t_batch(B, G_fixed, R_fixed, proto, frozen, t_s_fixed) for B in B_range]
    ax.plot(B_range, t, ls, color=color, linewidth=2, label=name)

ax.set_xlabel("B (batch_size)")
ax.set_ylabel("T_batch (seconds)")
ax.set_title(f"Protocol Comparison (G={G_fixed}, R={R_fixed}, t_s={t_s_fixed}s)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2: Model size sensitivity — T_batch vs t_s for each protocol
# ---------------------------------------------------------------------------

ax = axes[0, 1]
t_s_sweep = np.linspace(3, 30, 50)
B_fixed, G_fixed, R_fixed = 32, 4, 2

for name, proto, frozen, color, ls in configs:
    t = [t_batch(B_fixed, G_fixed, R_fixed, proto, frozen, ts) for ts in t_s_sweep]
    ax.plot(t_s_sweep, t, ls, color=color, linewidth=2, label=name)

ax.set_xlabel("t_s (seconds per debater turn — proxy for model size)")
ax.set_ylabel("T_batch (seconds)")
ax.set_title(f"Model Size Sensitivity (B={B_fixed}, G={G_fixed}, R={R_fixed})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Annotate model sizes
for ts, label in [(5, "~4B"), (10, "~8B"), (15, "~30B"), (25, "~120B")]:
    ax.axvline(x=ts, color="gray", linestyle=":", alpha=0.3)
    ax.text(ts, ax.get_ylim()[1] * 0.95, label, fontsize=7, ha="center", color="gray")

# ---------------------------------------------------------------------------
# Plot 3: Total run time (hours) for 50 batches — B×G heatmap, SEQ self-play
# ---------------------------------------------------------------------------

ax = axes[0, 2]
B_vals = [4, 8, 16, 32, 64]
G_vals = [2, 4, 8, 16]
N_batches = 50
t_s_fixed = 10

heatmap = np.zeros((len(G_vals), len(B_vals)))
for i, G in enumerate(G_vals):
    for j, B in enumerate(B_vals):
        heatmap[i, j] = total_hours(B, G, 2, "SEQ", False, t_s_fixed, N_batches)

im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn_r", origin="lower")
ax.set_xticks(range(len(B_vals)))
ax.set_xticklabels(B_vals)
ax.set_yticks(range(len(G_vals)))
ax.set_yticklabels(G_vals)
ax.set_xlabel("B (batch_size)")
ax.set_ylabel("G (group_size)")
ax.set_title(f"SEQ Self-Play R=2: {N_batches}-Batch Run (hours)")
for i in range(len(G_vals)):
    for j in range(len(B_vals)):
        ax.text(j, i, f"{heatmap[i,j]:.1f}h", ha="center", va="center", fontsize=8,
                color="white" if heatmap[i,j] > heatmap.max() * 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)

# ---------------------------------------------------------------------------
# Plot 4: Gradient efficiency per hour — sqrt(B) / T_batch
# ---------------------------------------------------------------------------

ax = axes[1, 0]
B_range = np.arange(4, 129, 4)

for name, proto, frozen, color, ls in configs:
    eff = [np.sqrt(B) / t_batch(B, 4, 2, proto, frozen, 10) * 3600 for B in B_range]
    ax.plot(B_range, eff, ls, color=color, linewidth=2, label=name)

ax.set_xlabel("B (batch_size, G=4)")
ax.set_ylabel("Learning efficiency (√B / T_batch × 3600)")
ax.set_title("Gradient Quality per Hour")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Mark peaks
for name, proto, frozen, color, ls in configs:
    eff = [np.sqrt(B) / t_batch(B, 4, 2, proto, frozen, 10) * 3600 for B in B_range]
    peak_idx = np.argmax(eff)
    ax.annotate(f"B={B_range[peak_idx]}", xy=(B_range[peak_idx], eff[peak_idx]),
                fontsize=7, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=color, lw=0.5),
                xytext=(B_range[peak_idx], eff[peak_idx] * 1.1))

# ---------------------------------------------------------------------------
# Plot 5: Rounds sensitivity — T_batch vs R for each protocol
# ---------------------------------------------------------------------------

ax = axes[1, 1]
R_range = np.arange(1, 6)
B_fixed, G_fixed = 32, 4

bar_width = 0.18
x = np.arange(len(R_range))

for idx, (name, proto, frozen, color, ls) in enumerate(configs):
    t = [t_batch(B_fixed, G_fixed, R, proto, frozen, 10) for R in R_range]
    ax.bar(x + idx * bar_width, t, bar_width, label=name, color=color,
           alpha=0.8 if ls == "-" else 0.5)

ax.set_xlabel("R (num_rounds)")
ax.set_ylabel("T_batch (seconds)")
ax.set_xticks(x + 1.5 * bar_width)
ax.set_xticklabels(R_range)
ax.set_title(f"Rounds Sensitivity (B={B_fixed}, G={G_fixed})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# ---------------------------------------------------------------------------
# Plot 6: Protocol speedup ratio (SIM/SEQ) across B·G
# ---------------------------------------------------------------------------

ax = axes[1, 2]
BG_range = np.arange(8, 1025, 8)

for frozen, ls, label_suffix in [(False, "-", "self-play"), (True, "--", "frozen-opp")]:
    ratios = []
    for bg in BG_range:
        B = max(4, bg // 4)
        G = 4
        t_seq = t_batch(B, G, 2, "SEQ", frozen, 10)
        t_sim = t_batch(B, G, 2, "SIM", frozen, 10)
        ratios.append(t_seq / t_sim)
    ax.plot(BG_range, ratios, ls, color="#9b59b6", linewidth=2,
            label=f"SEQ/SIM speedup ({label_suffix})")

ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.3)
ax.text(50, 2.05, "2× speedup", fontsize=7, color="gray")
ax.set_xlabel("B·G (total games per batch)")
ax.set_ylabel("Speedup ratio (SEQ time / SIM time)")
ax.set_title("SIMULTANEOUS Protocol Advantage (R=2)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 2.5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.tight_layout()
plt.savefig("scripts/debate_protocol_plots.png", dpi=150, bbox_inches="tight")
print("Saved to scripts/debate_protocol_plots.png")
plt.close()
