"""Figure: Strategy Evolution — rhetorical move rates over training (rung1)."""

import json
import re
import matplotlib.pyplot as plt
import numpy as np

# Load rung1 episodes
with open("logs/thinking-experiment/rung1-no-think/episodes/episodes.jsonl") as f:
    episodes = [json.loads(l) for l in f if l.strip()]

seen = set()
debates = []
for ep in episodes:
    did = ep["debate_id"]
    if did not in seen:
        seen.add(did)
        debates.append(ep)

n = len(debates)
q_size = n // 5

moves = {
    "Attack": ["flaw", "error", "incorrect", "wrong", "mistake", "contradicts", "fails"],
    "Evidence": ["evidence", "data shows", "studies show", "research", "experiment", "measured"],
    "Redirect": ["however", "the real question", "more importantly", "instead", "rather"],
    "Defend": ["i maintain", "my position", "i stand by", "my answer", "i argue"],
    "Concede": ["i concede", "you're right", "i agree with", "valid point", "fair point",
                "acknowledge", "correct to point out"],
}

# Compute per quintile
results = {move: [] for move in moves}

for qi in range(5):
    chunk = debates[qi * q_size:(qi + 1) * q_size]
    for move_name, keywords in moves.items():
        count = 0
        total = 0
        for d in chunk:
            for turn in d.get("transcript", []):
                if turn.get("phase") != "critique":
                    continue
                text = turn.get("text", "").lower()
                total += 1
                if any(kw in text for kw in keywords):
                    count += 1
        results[move_name].append(100 * count / total if total else 0)

# Plot: grouped by move, 5 bars (quintiles) per group
fig, ax = plt.subplots(figsize=(7.4, 4.6))

move_names = list(moves.keys())
n_moves = len(move_names)
n_q = 5
x = np.arange(n_moves)
width = 0.14
cividis = plt.cm.cividis(np.linspace(0.15, 0.85, n_q))

for qi in range(n_q):
    vals = [results[m][qi] for m in move_names]
    offset = (qi - 2) * width
    bars = ax.bar(x + offset, vals, width, color=cividis[qi], edgecolor="#404040",
                  linewidth=0.4, label=f"Q{qi + 1}")
    # Label Q1 and Q5 for collapsing moves
    if qi in [0, 4]:
        for i, (m, v) in enumerate(zip(move_names, vals)):
            if m in ["Redirect", "Defend", "Concede"]:
                ax.text(x[i] + offset, v + 1.5, f"{v:.0f}%",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                        color="#404040")

# Ceiling band
ax.axhspan(90, 102, color="#F2F2F2", zorder=0)
ax.text(0.5, 96, "ceiling strategies", ha="center", va="center",
        fontsize=8, color="#AAAAAA", style="italic")

# Collapse annotation
ax.annotate("Interactive moves\ncollapse by late training",
            xy=(3.2, 20), xytext=(3.5, 55),
            fontsize=9, fontweight="bold", color="#D55E00",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2),
            ha="center")

ax.set_xticks(x)
ax.set_xticklabels(move_names, fontsize=10)
ax.set_ylim(0, 102)
ax.set_ylabel("Critique responses containing move (%)", fontsize=10)
ax.set_title("Attack + Evidence stay at ceiling;\ninteractive moves collapse",
             fontsize=12, fontweight="bold", color="#595959", pad=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5,
          fontsize=9, frameon=False, title="Training quintile", title_fontsize=9)

plt.tight_layout()
plt.savefig("reports/figures/fig_strategy_evolution.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_strategy_evolution.png")
