"""Generate seat bias + judge analysis figures (Task #2)."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colorblind-safe palette (Okabe-Ito)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN = "#56B4E9"
C_GRAY = "#999999"

OUT = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

DATA_R1 = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/thinking-experiment/rung1-no-think/episodes/episodes.jsonl")
DATA_R2 = Path("/Users/joalja/Documents/Github/ext/tinker-cookbook/logs/thinking-experiment/rung2-private-think/episodes/episodes.jsonl")


def load_debates(path):
    eps = []
    with open(path) as f:
        for line in f:
            eps.append(json.loads(line))
    return [e for e in eps if e["role"] == "debater_a"]


def quintiles(debates, n_q=5):
    q_size = len(debates) // n_q
    return [debates[i * q_size : (i + 1) * q_size] for i in range(n_q)]


def sig(ep, key, default=0.0):
    return ep["signals"].get(key, default)


debates_r1 = load_debates(DATA_R1)
debates_r2 = load_debates(DATA_R2)
quints_r1 = quintiles(debates_r1)
quints_r2 = quintiles(debates_r2)

print(f"Rung1: {len(debates_r1)} debates, Rung2: {len(debates_r2)} debates")

# ── Fig 1: Concession Dynamics ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = [f"Q{i+1}" for i in range(5)]
x = np.arange(5)
w = 0.35

sc_a = [np.mean([sig(e, "stance_change.debater_a") for e in q]) for q in quints_r1]
sc_b = [np.mean([sig(e, "stance_change.debater_b") for e in q]) for q in quints_r1]

bars_a = ax.bar(x - w / 2, sc_a, w, label="Seat A concedes", color=C_BLUE, edgecolor="white", linewidth=0.5)
bars_b = ax.bar(x + w / 2, sc_b, w, label="Seat B concedes", color=C_ORANGE, edgecolor="white", linewidth=0.5)

for bars in [bars_a, bars_b]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Training quintile")
ax.set_ylabel("Stance change rate")
ax.set_title("Concession dynamics over training")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(max(sc_a), max(sc_b)) * 1.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_concession_dynamics.png")
plt.close(fig)
print(f"[1/7] fig_concession_dynamics.png — A: {sc_a}, B: {sc_b}")

# ── Fig 2: Verdict Categories (B wins) ────────────────────────────────
categories = [
    "A initially wrong, corrected",
    "B's reasoning had fewer logical errors",
    "Both correct, B marginally better",
    "B's explanation clearer/more precise",
    "B reached correct conclusion earlier",
]
values = [36.7, 23.3, 20.0, 13.3, 6.7]
colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE]

fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(categories))
bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=10)
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.set_xlabel("% of B-win verdicts")
ax.set_title("Judge verdict categories when B wins")
ax.set_xlim(0, 45)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "fig_verdict_categories.png")
plt.close(fig)
print("[2/7] fig_verdict_categories.png")

# ── Fig 3: Win Rate by Condition ───────────────────────────────────────
decisive = [e for e in debates_r1 if e["winner"] in ("debater_a", "debater_b")]

def b_win_rate(subset):
    if not subset:
        return 0
    return sum(1 for e in subset if e["winner"] == "debater_b") / len(subset) * 100

# Overall
wr_overall = b_win_rate(decisive)

# Same answer
same_ans = [e for e in decisive if e["answers"]["public_debater_a"] == e["answers"]["public_debater_b"]]
wr_same = b_win_rate(same_ans)

# Same answer + no concession
same_no_conc = [e for e in same_ans
                if sig(e, "stance_change.debater_a") == 0 and sig(e, "stance_change.debater_b") == 0]
wr_same_nc = b_win_rate(same_no_conc)

# A concedes to B
a_conc = [e for e in decisive if sig(e, "stance_change.debater_a") == 1]
wr_a_conc = b_win_rate(a_conc)

# B concedes to A
b_conc = [e for e in decisive if sig(e, "stance_change.debater_b") == 1]
wr_b_conc = b_win_rate(b_conc)

conditions = ["Overall", "Same answer", "Same answer\n+ no concession", "A concedes\nto B", "B concedes\nto A"]
wr_vals = [wr_overall, wr_same, wr_same_nc, wr_a_conc, wr_b_conc]
cond_colors = [C_GRAY, C_BLUE, C_CYAN, C_ORANGE, C_GREEN]
cond_n = [len(decisive), len(same_ans), len(same_no_conc), len(a_conc), len(b_conc)]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(conditions))
bars = ax.bar(x, wr_vals, color=cond_colors, edgecolor="white", linewidth=0.5, width=0.65)
for bar, v, n in zip(bars, wr_vals, cond_n):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

ax.axhline(50, color="black", linestyle=":", alpha=0.4, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("B win rate (% of decisive)")
ax.set_title("Seat B win rate by condition")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(OUT / "fig_win_rate_by_condition.png")
plt.close(fig)
print(f"[3/7] fig_win_rate_by_condition.png — vals: {[f'{v:.1f}' for v in wr_vals]}")

# ── Fig 4: Truth Premium ──────────────────────────────────────────────
# P(win|correct) vs P(win|wrong) for each seat
a_correct = [e for e in decisive if sig(e, "accuracy.debater_a") == 1]
a_wrong = [e for e in decisive if sig(e, "accuracy.debater_a") == 0]
b_correct = [e for e in decisive if sig(e, "accuracy.debater_b") == 1]
b_wrong = [e for e in decisive if sig(e, "accuracy.debater_b") == 0]

p_a_win_correct = sum(1 for e in a_correct if e["winner"] == "debater_a") / len(a_correct) * 100 if a_correct else 0
p_a_win_wrong = sum(1 for e in a_wrong if e["winner"] == "debater_a") / len(a_wrong) * 100 if a_wrong else 0
p_b_win_correct = sum(1 for e in b_correct if e["winner"] == "debater_b") / len(b_correct) * 100 if b_correct else 0
p_b_win_wrong = sum(1 for e in b_wrong if e["winner"] == "debater_b") / len(b_wrong) * 100 if b_wrong else 0

fig, ax = plt.subplots(figsize=(6, 4.5))
x = np.arange(2)
w = 0.35
bars1 = ax.bar(x - w / 2, [p_a_win_correct, p_b_win_correct], w,
               label="P(win | correct)", color=C_GREEN, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w / 2, [p_a_win_wrong, p_b_win_wrong], w,
               label="P(win | wrong)", color=C_RED, edgecolor="white", linewidth=0.5)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=10)

# Annotate truth premium
delta_a = p_a_win_correct - p_a_win_wrong
delta_b = p_b_win_correct - p_b_win_wrong
ax.annotate(f"{delta_a:+.1f}pp", xy=(0, max(p_a_win_correct, p_a_win_wrong) + 6),
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold")
ax.annotate(f"{delta_b:+.1f}pp", xy=(1, max(p_b_win_correct, p_b_win_wrong) + 10),
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Seat A", "Seat B"])
ax.set_ylabel("Win rate (%)")
ax.set_title("Truth premium: correct vs wrong answer")
ax.set_ylim(0, 110)
ax.legend(loc="center left")
fig.tight_layout()
fig.savefig(OUT / "fig_truth_premium.png")
plt.close(fig)
print(f"[4/7] fig_truth_premium.png — A: {delta_a:.1f}pp, B: {delta_b:.1f}pp")

# ── Fig 5: Sycophancy Trend ───────────────────────────────────────────
def syc_rate(chunk):
    conc_a = [e for e in chunk if sig(e, "stance_change.debater_a") == 1]
    conc_b = [e for e in chunk if sig(e, "stance_change.debater_b") == 1]
    all_conc = len(conc_a) + len(conc_b)
    if all_conc == 0:
        return 0
    syc_a = sum(1 for e in conc_a if sig(e, "accuracy.debater_b") == 0)
    syc_b = sum(1 for e in conc_b if sig(e, "accuracy.debater_a") == 0)
    return (syc_a + syc_b) / all_conc

syc_r1 = [syc_rate(q) for q in quints_r1]
syc_r2 = [syc_rate(q) for q in quints_r2]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(5)
ax.plot(x, syc_r1, "o-", color=C_RED, linewidth=2, markersize=8, label="Rung 1 (no thinking)")
ax.plot(x, syc_r2, "s--", color=C_BLUE, linewidth=2, markersize=8, label="Rung 2 (private thinking)")

for i in range(5):
    gap = abs(syc_r1[i] - syc_r2[i])
    if gap < 0.06:
        # Labels would collide — push apart
        upper = max(syc_r1[i], syc_r2[i])
        lower = min(syc_r1[i], syc_r2[i])
        r1_offset = 0.035 if syc_r1[i] >= syc_r2[i] else -0.045
        r2_offset = 0.035 if syc_r2[i] > syc_r1[i] else -0.045
    else:
        r1_offset = 0.02
        r2_offset = -0.035
    ax.text(x[i], syc_r1[i] + r1_offset, f"{syc_r1[i]:.2f}", ha="center", fontsize=9, color=C_RED)
    ax.text(x[i], syc_r2[i] + r2_offset, f"{syc_r2[i]:.2f}", ha="center", fontsize=9, color=C_BLUE)

ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
ax.set_xlabel("Training quintile")
ax.set_ylabel("Sycophancy rate\n(concedes to wrong / all concessions)")
ax.set_title("Sycophancy rate over training")
ax.set_ylim(0.4, 1.0)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_sycophancy_trend.png")
plt.close(fig)
print(f"[5/7] fig_sycophancy_trend.png — R1: {syc_r1}, R2: {syc_r2}")

# ── Fig 6: Bullshit Contest Rate ───────────────────────────────────────
def bs_rate(chunk):
    dec = [e for e in chunk if e["winner"] in ("debater_a", "debater_b")]
    if not dec:
        return 0
    both_wrong = sum(1 for e in dec
                     if sig(e, "accuracy.debater_a") == 0 and sig(e, "accuracy.debater_b") == 0)
    return both_wrong / len(dec)

bs_r1 = [bs_rate(q) for q in quints_r1]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(5)
ax.plot(x, bs_r1, "o-", color=C_RED, linewidth=2, markersize=8)
ax.fill_between(x, bs_r1, alpha=0.15, color=C_RED)

for i in range(5):
    ax.text(x[i], bs_r1[i] + 0.012, f"{bs_r1[i]:.2f}", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
ax.set_xlabel("Training quintile")
ax.set_ylabel("Both-wrong decisive rate")
ax.set_title("Bullshit contest rate over training")
ax.set_ylim(0, max(bs_r1) * 1.3)
fig.tight_layout()
fig.savefig(OUT / "fig_bullshit_contest.png")
plt.close(fig)
print(f"[6/7] fig_bullshit_contest.png — {bs_r1}")

# ── Fig 7: Wrong Wins Tactics ─────────────────────────────────────────
tactics = [
    "Style on same answer",
    "Punishing self-correction",
    "Strawmanning",
    "Exploiting local errors",
    "Precision disputes",
    "Other",
]
tactic_pcts = [29, 24, 18, 18, 6, 5]
tactic_colors = [C_BLUE, C_ORANGE, C_RED, C_GREEN, C_PURPLE, C_GRAY]

fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    tactic_pcts,
    labels=tactics,
    colors=tactic_colors,
    autopct="%1.0f%%",
    startangle=90,
    pctdistance=0.8,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
)
for t in autotexts:
    t.set_fontsize(10)
    t.set_fontweight("bold")
for t in texts:
    t.set_fontsize(10)
ax.set_title("B's winning tactics when wrong (donut)")
fig.tight_layout()
fig.savefig(OUT / "fig_wrong_wins_tactics.png")
plt.close(fig)
print("[7/7] fig_wrong_wins_tactics.png")

print("\nAll 7 figures saved to", OUT)
