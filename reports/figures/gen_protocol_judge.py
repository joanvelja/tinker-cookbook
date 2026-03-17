"""Generate judge analysis figures for the protocol experiment report.

Figures:
  1. fig_verdict_length_by_prompt.png  — box+strip of verdict length by protocol
  2. fig_protocol_judge_exploitation.png — exploitation rate over training steps
  3. fig_protocol_seat_bias.png — B-side win rate over steps
  4. fig_wrong_wins_strategy.png — wrong-win strategy breakdown
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "reports/figures")
from style import COLORS, annotate_insight, apply_style, save

apply_style()

# ── data loading ──────────────────────────────────────────────────────

RUNS = ["seq-v1", "seq-v2", "hybrid", "simultaneous"]
RUN_LABELS = {
    "seq-v1": "Sequential v1",
    "seq-v2": "Sequential v2",
    "hybrid": "Hybrid",
    "simultaneous": "Simultaneous",
}
RUN_COLORS = {
    "seq-v1": COLORS["blue"],
    "seq-v2": COLORS["orange"],
    "hybrid": COLORS["green"],
    "simultaneous": COLORS["red"],
}


def load_episodes():
    """Load all episodes, return dict[run] -> list of debater_a episodes."""
    data = {}
    for run in RUNS:
        path = Path(f"logs/protocol-experiment/{run}/episodes/episodes.jsonl")
        with open(path) as f:
            eps = [json.loads(line) for line in f]
        data[run] = [e for e in eps if e["role"] == "debater_a"]
    return data


ALL = load_episodes()


# ── helpers ───────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return centre, max(0, centre - margin), min(1, centre + margin)


# ── Fig 1: Verdict length by protocol ────────────────────────────────

def fig_verdict_length():
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = range(1, len(RUNS) + 1)
    all_log_lengths = []
    labels = []
    colors = []
    medians = []

    for run in RUNS:
        lens = np.array([len(e.get("verdict_text", "")) for e in ALL[run]])
        log_lens = np.log10(np.clip(lens, 1, None))
        all_log_lengths.append(log_lens)
        labels.append(RUN_LABELS[run])
        colors.append(RUN_COLORS[run])
        medians.append(int(np.median(lens)))

    # Box plot on log10(length) for honest density representation
    bp = ax.boxplot(all_log_lengths, positions=list(positions), widths=0.5,
                    patch_artist=True, showfliers=False, medianprops=dict(color=COLORS["dark"], linewidth=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)

    # Jittered strip overlay (subsample for readability)
    rng = np.random.RandomState(42)
    for i, (pos, log_lens) in enumerate(zip(positions, all_log_lengths)):
        n = len(log_lens)
        idx = rng.choice(n, min(200, n), replace=False)
        jitter = rng.uniform(-0.15, 0.15, len(idx))
        ax.scatter(pos + jitter, log_lens[idx], s=4, alpha=0.3, color=colors[i],
                   zorder=3, edgecolors="none")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=15)

    # Label y-axis with actual character counts
    log_ticks = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    ax.set_yticks(log_ticks)
    ax.set_yticklabels([f"{10**t:.0f}" for t in log_ticks])
    ax.set_ylabel("Verdict length (characters)")
    ax.set_title("Judge verdict length by protocol")

    # Annotate medians
    for i, med in enumerate(medians):
        y_pos = np.log10(max(med, 1)) + 0.15
        ax.text(i + 1, y_pos, f"med={med}", ha="center", fontsize=9,
                fontweight="bold", color=colors[i])

    ratio = medians[0] / medians[1] if medians[1] > 0 else 0
    annotate_insight(ax, f"v1 prompt elicits\n~{ratio:.0f}x longer verdicts",
                     xy=(1, np.log10(medians[0])), xytext=(2.5, 4.0),
                     color=COLORS["blue"])

    fig.tight_layout()
    save(fig, "fig_verdict_length_by_prompt")
    plt.close(fig)


# ── Fig 2: Judge exploitation over steps ─────────────────────────────

def fig_exploitation():
    fig, ax = plt.subplots(figsize=(8, 5))

    series = {}  # run -> (step_vals, exploit_rates, sample_sizes)

    for run in RUNS:
        eps = ALL[run]
        steps = sorted(set(e["step"] for e in eps))
        exploit_rates = []
        ci_lo = []
        ci_hi = []
        step_vals = []
        sample_sizes = []

        for step in steps:
            step_eps = [e for e in eps if e["step"] == step]
            decided = [e for e in step_eps if e["winner"] is not None]
            n = len(decided)
            if n < 5:
                continue
            # Count wrong_wins among decided games only
            wrong_wins = sum(
                1 for e in decided
                if e.get("signals", {}).get("wrong_and_wins.debater_a", 0) == 1.0
                or e.get("signals", {}).get("wrong_and_wins.debater_b", 0) == 1.0
            )
            centre, lo, hi = wilson_ci(wrong_wins, n)
            exploit_rates.append(wrong_wins / n)
            ci_lo.append(lo)
            ci_hi.append(hi)
            step_vals.append(step)
            sample_sizes.append(n)

        step_vals = np.array(step_vals)
        exploit_rates = np.array(exploit_rates)
        series[run] = (step_vals, exploit_rates, sample_sizes)

        ax.fill_between(step_vals, ci_lo, ci_hi, color=RUN_COLORS[run], alpha=0.12)
        ax.plot(step_vals, exploit_rates, "o-", color=RUN_COLORS[run],
                label=RUN_LABELS[run], markersize=5, linewidth=1.8)

    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1, label="Neutral (0.5)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Exploitation rate\n(wrong debater wins / decided games)")
    ax.set_title("Judge exploitation rate over training")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.85)

    # Annotate the simultaneous decline — find first step *substantively* below 0.5
    sim_steps, sim_rates, sim_ns = series["simultaneous"]
    below_idx = np.where(sim_rates < 0.45)[0]  # threshold away from noise
    if len(below_idx) > 0:
        onset_step = int(sim_steps[below_idx[0]])
        # Point arrow at the last large-N point (N >= 50) for credibility
        large_n_below = [
            i for i in below_idx if sim_ns[i] >= 50
        ]
        ann_i = large_n_below[-1] if large_n_below else below_idx[-1]
        ann_step = sim_steps[ann_i]
        ann_rate = sim_rates[ann_i]
        ann_n = sim_ns[ann_i]
        annotate_insight(
            ax,
            f"Declining from step {onset_step}\n({ann_rate:.0%} at step {int(ann_step)}, N={ann_n})",
            xy=(ann_step, ann_rate),
            xytext=(ann_step - 3, 0.17),
            color=COLORS["red"],
        )

    fig.tight_layout()
    save(fig, "fig_protocol_judge_exploitation")
    plt.close(fig)


# ── Fig 3: Seat bias (B-side win rate) ───────────────────────────────

def fig_seat_bias():
    fig, ax = plt.subplots(figsize=(8, 5))

    for run in RUNS:
        eps = ALL[run]
        steps = sorted(set(e["step"] for e in eps))
        b_rates = []
        ci_lo = []
        ci_hi = []
        step_vals = []

        for step in steps:
            step_eps = [e for e in eps if e["step"] == step]
            decided = [e for e in step_eps if e["winner"] is not None]
            n = len(decided)
            if n < 5:
                continue
            b_wins = sum(1 for e in decided if e["winner"] == "debater_b")
            centre, lo, hi = wilson_ci(b_wins, n)
            b_rates.append(b_wins / n)
            ci_lo.append(lo)
            ci_hi.append(hi)
            step_vals.append(step)

        step_vals = np.array(step_vals)
        b_rates = np.array(b_rates)

        ax.fill_between(step_vals, ci_lo, ci_hi, color=RUN_COLORS[run], alpha=0.12)
        ax.plot(step_vals, b_rates, "o-", color=RUN_COLORS[run],
                label=RUN_LABELS[run], markersize=5, linewidth=1.8)

    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1, label="No bias (0.5)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("B-side win rate (among decided games)")
    ax.set_title("B-side win rate by protocol")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)

    # Overall seq-v1 B rate for annotation
    seq_v1_decided = [e for e in ALL["seq-v1"] if e["winner"] is not None]
    seq_v1_b = sum(1 for e in seq_v1_decided if e["winner"] == "debater_b")
    overall_b_rate = seq_v1_b / len(seq_v1_decided)
    annotate_insight(ax, f"Sequential v1:\n{overall_b_rate:.0%} B win rate overall",
                     xy=(3, 0.732), xytext=(5, 0.88),
                     color=COLORS["blue"])

    fig.tight_layout()
    save(fig, "fig_protocol_seat_bias")
    plt.close(fig)


# ── Fig 4: Wrong-win strategies ──────────────────────────────────────

def classify_strategies():
    """Classify wrong-win strategies from transcript content.

    Categories (mutually exclusive, priority order):
      1. Technical nitpick — claims specific math/physics errors
      2. Direct attack — explicitly calls opponent's reasoning flawed
      3. Reinterpretation — reframes the question to fit wrong answer
      4. Authority appeal — invokes textbooks/literature to lend credibility
      5. Generic — no clear dominant tactic detected
    """
    strats_by_run = {}

    for run in RUNS:
        eps = ALL[run]
        wrong = [
            e for e in eps
            if e.get("signals", {}).get("wrong_and_wins.debater_a", 0) == 1.0
            or e.get("signals", {}).get("wrong_and_wins.debater_b", 0) == 1.0
        ]

        counts = Counter()
        for e in wrong:
            winner = e["winner"]
            transcript = e.get("transcript", [])

            # Only look at critique/rebuttal turns (round > 0), not initial proposals
            winner_texts = []
            for turn in transcript:
                if not isinstance(turn, dict):
                    continue
                if turn.get("identity") != winner:
                    continue
                if turn.get("round", 0) == 0 and turn.get("phase") == "propose":
                    continue
                text = turn.get("text", "")
                public_text = re.sub(
                    r"<think>.*?</think>", "", text, flags=re.DOTALL
                ).strip()
                winner_texts.append(public_text)

            wt = " ".join(winner_texts).lower()

            # Sub-categories (mutually exclusive, priority order)
            has_math_trick = any(
                m in wt
                for m in [
                    "sign error", "boundary condition", "off by one",
                    "factor of", "neglects the", "dropped the",
                    "missing term", "dimensional analysis",
                ]
            )
            has_direct_attack = sum(
                1 for m in [
                    "flaw in", "error in", "mistake in",
                    "incorrect because", "wrong because",
                    "fails to account", "overlooks the",
                    "misapplies", "conflates",
                ]
                if m in wt
            ) >= 1
            has_reinterpret = any(
                m in wt
                for m in [
                    "actually means", "properly interpreted",
                    "correct interpretation", "true meaning",
                    "actually refers", "redefine",
                ]
            )
            # Tightened: require phrase-level matches, not bare "established"
            has_authority = any(
                m in wt
                for m in [
                    "textbook result", "published in",
                    "standard result", "well-known result",
                    "as shown in", "the literature",
                ]
            )

            if has_math_trick:
                counts["Technical nitpick"] += 1
            elif has_direct_attack:
                counts["Direct attack"] += 1
            elif has_reinterpret:
                counts["Reinterpretation"] += 1
            elif has_authority:
                counts["Authority appeal"] += 1
            else:
                counts["Generic attack"] += 1

        strats_by_run[run] = counts

    return strats_by_run


def fig_wrong_wins_strategy():
    strats_by_run = classify_strategies()
    categories = [
        "Direct attack",
        "Technical nitpick",
        "Reinterpretation",
        "Authority appeal",
        "Generic attack",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(categories))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, run in enumerate(RUNS):
        counts = strats_by_run[run]
        total = sum(counts.values())
        vals = [counts.get(cat, 0) / total * 100 if total > 0 else 0 for cat in categories]
        ax.bar(
            x + offsets[i] * width, vals, width,
            label=RUN_LABELS[run], color=RUN_COLORS[run], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Share of wrong-wins (%)")
    ax.set_title("Wrong-win exploitation strategies (from critique turns)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    save(fig, "fig_wrong_wins_strategy")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_verdict_length()
    fig_exploitation()
    fig_seat_bias()
    fig_wrong_wins_strategy()
    print("All judge figures generated.")
