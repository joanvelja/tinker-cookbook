"""Strategy Evolution — Q1 vs Q5 paired bars only. Cleaner redesign."""

import json
import re
import matplotlib.pyplot as plt
import numpy as np

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
early = debates[:q_size]
late = debates[4 * q_size:]

moves = {
    "Attack": ["flaw", "error", "incorrect", "wrong", "mistake", "contradicts", "fails"],
    "Evidence": ["evidence", "data shows", "studies show", "research", "experiment", "measured"],
    "Redirect": ["however", "the real question", "more importantly", "instead", "rather"],
    "Defend": ["i maintain", "my position", "i stand by", "my answer", "i argue"],
    "Concede": ["i concede", "you're right", "i agree with", "valid point", "fair point",
                "acknowledge", "correct to point out"],
}

def compute_rates(chunk):
    rates = {}
    for move_name, keywords in moves.items():
        count = total = 0
        for d in chunk:
            for turn in d.get("transcript", []):
                if turn.get("phase") != "critique":
                    continue
                text = turn.get("text", "").lower()
                total += 1
                if any(kw in text for kw in keywords):
                    count += 1
        rates[move_name] = 100 * count / total if total else 0
    return rates

early_rates = compute_rates(early)
late_rates = compute_rates(late)

# Sort by delta (biggest collapse first)
move_names = sorted(moves.keys(), key=lambda m: early_rates[m] - late_rates[m], reverse=True)

fig, ax = plt.subplots(figsize=(7.5, 4))

x = np.arange(len(move_names))
width = 0.32

bars_early = ax.bar(x - width / 2, [early_rates[m] for m in move_names], width,
                     color="#4E79A7", edgecolor="#404040", linewidth=0.5, label="Q1 (early)")
bars_late = ax.bar(x + width / 2, [late_rates[m] for m in move_names], width,
                    color="#F28E2B", edgecolor="#404040", linewidth=0.5, label="Q5 (late)")

# Value labels on each bar
for bars, rates_dict in [(bars_early, early_rates), (bars_late, late_rates)]:
    for bar, name in zip(bars, move_names):
        val = rates_dict[name]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#404040")

# Delta annotations between each pair
for i, name in enumerate(move_names):
    delta = late_rates[name] - early_rates[name]
    sign = "+" if delta > 0 else ""
    color = "#59A14F" if delta > 0 else "#D55E00"
    mid_y = max(early_rates[name], late_rates[name]) + 8
    ax.text(x[i], mid_y, f"{sign}{delta:.0f}pp",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

ax.set_xticks(x)
ax.set_xticklabels(move_names, fontsize=10)
ax.set_ylim(0, 110)
ax.set_ylabel("Responses containing move (%)", fontsize=10)
ax.set_title("Attack + Evidence persist; interactive moves collapse",
             fontsize=12, fontweight="bold", color="#595959", pad=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.6)
ax.set_axisbelow(True)

ax.legend(loc="center right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#DDDDDD")

plt.tight_layout()
plt.savefig("reports/figures/fig_strategy_evolution.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_strategy_evolution.png")
