"""Figure: Think Block Corruption — 4-panel degradation over training (rung2)."""

import json
import re
import matplotlib.pyplot as plt
import numpy as np

# Load rung2 episodes
with open("logs/thinking-experiment/rung2-private-think/episodes/episodes.jsonl") as f:
    episodes = [json.loads(l) for l in f if l.strip()]

# Deduplicate
seen = set()
debates = []
for ep in episodes:
    did = ep["debate_id"]
    if did not in seen:
        seen.add(did)
        debates.append(ep)

n = len(debates)
q_size = n // 5

# Compute metrics per quintile
think_lengths = []
template_rates = []
citation_rates = []
stands_rates = []

for qi in range(5):
    chunk = debates[qi * q_size:(qi + 1) * q_size]
    lengths = []
    template_count = 0
    cite_count = 0
    stands_count = 0
    total_thinks = 0

    for d in chunk:
        for turn in d.get("transcript", []):
            text = turn.get("text", "")
            think_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
            if not think_match:
                continue
            think = think_match.group(1)
            total_thinks += 1
            lengths.append(len(think))

            # Template: starts with numbered pattern
            if re.match(r"\s*(1[\.\):]|\*\*1)", think.strip()):
                template_count += 1

            # Citations
            if re.search(r"et al\.|Journal of|Nature |Science |Phys\. Rev|Chem\. Rev|PNAS|Nat\. Commun", think):
                cite_count += 1

            # "Answer stands"
            last_500 = think[-500:].lower()
            if any(p in last_500 for p in ["my answer stands", "i maintain", "no change needed",
                                            "i do not change", "answer remains"]):
                stands_count += 1

    think_lengths.append(np.mean(lengths) if lengths else 0)
    template_rates.append(100 * template_count / total_thinks if total_thinks else 0)
    citation_rates.append(100 * cite_count / total_thinks if total_thinks else 0)
    stands_rates.append(100 * stands_count / total_thinks if total_thinks else 0)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
fig.suptitle("Private reasoning grows longer, looser, and less faithful",
             fontsize=13, fontweight="bold", color="#595959", y=0.98)

x = np.arange(5)
labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
width = 0.55

panels = [
    (axes[0, 0], think_lengths, "#4E79A7", "A. Think block length (chars)", None, 0),
    (axes[0, 1], template_rates, "#76B7B2", "B. Formulaic template rate (%)", 100, 0),
    (axes[1, 0], citation_rates, "#D55E00", "C. Fabricated citation rate (%)", None, 0),
    (axes[1, 1], stands_rates, "#B07AA1", 'D. "Answer stands" conclusion (%)', None, 0),
]

for ax, data, color, title, ylim_max, ylim_min in panels:
    bars = ax.bar(x, data, width, color=color, edgecolor="#404040", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#595959", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)

    if ylim_max:
        ax.set_ylim(ylim_min, ylim_max)

    # Label first and last bars
    for i in [0, 4]:
        val = data[i]
        fmt = f"{val:.0f}" if val > 10 else f"{val:.1f}"
        ax.text(x[i], val + (max(data) * 0.03), fmt,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#404040")

    # Delta annotation
    delta_pct = (data[4] - data[0]) / data[0] * 100 if data[0] > 0 else 0
    sign = "+" if delta_pct > 0 else ""
    ax.text(0.95, 0.92, f"{sign}{delta_pct:.0f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            color="#D55E00" if delta_pct > 0 else "#59A14F")

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("reports/figures/fig_think_block_corruption.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_think_block_corruption.png")
