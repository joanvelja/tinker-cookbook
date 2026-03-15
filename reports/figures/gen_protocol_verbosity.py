"""Verbosity and quality figures for the protocol comparison report.

Generates:
  1. fig_protocol_token_growth.png — ac_tokens_per_turn AND growth rate, 4 runs
  2. fig_protocol_parse_success.png — parse_success over steps, death threshold
  3. fig_protocol_verbosity_anatomy.png — content breakdown early vs late (simultaneous)
  4. fig_accuracy_vs_length.png — accuracy vs word count scatter
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "reports/figures")
from style import COLORS, annotate_insight, apply_style, save

apply_style()

BASE = Path("logs/protocol-experiment")
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
RUN_MARKERS = {"seq-v1": "o", "seq-v2": "s", "hybrid": "D", "simultaneous": "^"}


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_metrics(run):
    with open(BASE / run / "metrics.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_episodes(run):
    with open(BASE / run / "episodes" / "episodes.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Figure 1: Token Growth ──────────────────────────────────────────────────


def fig_token_growth():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35)

    for run in RUNS:
        rows = load_metrics(run)
        tokens = [r.get("env/all/ac_tokens_per_turn", 0) or 0 for r in rows]
        steps = np.arange(len(tokens))
        c = RUN_COLORS[run]
        m = RUN_MARKERS[run]

        # Left panel: raw tokens per turn
        ax1.plot(
            steps, tokens, f"{m}-", color=c, markersize=5, linewidth=1.6,
            label=RUN_LABELS[run],
        )

        # Right panel: growth rate (% relative to step 0)
        baseline = tokens[0] if tokens[0] > 0 else 1
        growth = [(t / baseline - 1) * 100 for t in tokens]
        ax2.plot(
            steps, growth, f"{m}-", color=c, markersize=5, linewidth=1.6,
            label=RUN_LABELS[run],
        )

    ax1.set_title("Tokens per Turn (action tokens)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Tokens / turn")

    ax2.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_title("Cumulative Change vs Step 0")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Growth (%)")

    ax1.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Response length grows fastest under simultaneous protocol",
        fontsize=13, fontweight="bold", color=COLORS["dark"], y=1.02,
    )

    save(fig, "fig_protocol_token_growth")
    plt.close(fig)


# ── Figure 2: Parse Success ─────────────────────────────────────────────────


def fig_parse_success():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for run in RUNS:
        rows = load_metrics(run)
        ps = [r.get("env/all/parse_success", 1.0) for r in rows]
        steps = np.arange(len(ps))
        ax.plot(
            steps, ps, f"{RUN_MARKERS[run]}-", color=RUN_COLORS[run],
            markersize=5, linewidth=1.6, label=RUN_LABELS[run],
        )

    # Death threshold line
    ax.axhline(0.5, color=COLORS["accent"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(
        0.3, 0.48, "death threshold", fontsize=8.5, color=COLORS["accent"],
        fontstyle="italic", transform=ax.get_yaxis_transform(),
    )

    ax.set_title("Parse Success Rate Over Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Parse success rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=8.5)

    # Check if simultaneous actually dies
    sim_rows = load_metrics("simultaneous")
    sim_ps = [r.get("env/all/parse_success", 1.0) for r in sim_rows]
    min_ps = min(sim_ps)
    min_step = sim_ps.index(min_ps)
    if min_ps < 0.7:
        annotate_insight(
            ax, f"Simultaneous drops\nto {min_ps:.0%} at step {min_step}",
            xy=(min_step, min_ps), xytext=(min_step + 1.5, min_ps + 0.15),
        )

    fig.suptitle(
        "Simultaneous protocol loses ability to produce valid answers",
        fontsize=13, fontweight="bold", color=COLORS["dark"], y=1.02,
    )

    save(fig, "fig_protocol_parse_success")
    plt.close(fig)


# ── Figure 3: Verbosity Anatomy (simultaneous early vs late) ────────────────


BACKTRACK_PAT = re.compile(
    r"(?i)\b(wait[,!.\s]|let me re(?:consider|check|verify|calculate|examine)|"
    r"actually,?\s|hold on|re-verify|re-check|reconsider|"
    r"let me recalculate|no,?\s+that(?:'s| is) (?:wrong|incorrect|not)|"
    r"hmm,?\s|on second thought|correction:|I made an? (?:error|mistake))"
)

META_PAT = re.compile(
    r"(?i)(in (?:my|this) (?:previous|earlier|first)|"
    r"as (?:I|we) (?:noted|discussed|mentioned)|"
    r"to summarize|in summary|in conclusion|"
    r"moving on to|turning (?:now )?to|"
    r"the (?:other|alternative) (?:expert|response|answer))"
)


def _count_categories(text):
    """Classify words into content categories using phrase-level detection.

    Counts phrase occurrences (not whole lines) to avoid inflating categories
    when a single trigger word appears in a long substantive line.
    Returns dict with counts for each category.
    """
    total_words = len(text.split())
    lines = text.split("\n")

    # Structural scaffolding: count words in header and bullet lines
    scaffolding_words = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,4}\s", stripped):
            scaffolding_words += len(stripped.split())
        elif re.match(r"^[\*\-]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            scaffolding_words += len(stripped.split())

    # Backtrack and meta: count phrase occurrences (not line-level)
    backtrack_phrases = len(BACKTRACK_PAT.findall(text))
    meta_phrases = len(META_PAT.findall(text))

    substantive_words = max(0, total_words - scaffolding_words)

    return {
        "substantive": substantive_words,
        "backtrack_phrases": backtrack_phrases,
        "scaffolding": scaffolding_words,
        "meta_phrases": meta_phrases,
    }


def fig_verbosity_anatomy():
    episodes = load_episodes("simultaneous")
    train_eps = [e for e in episodes if e["split"] == "train"]

    # Early (steps 0-1) vs Late (steps 5+)
    early_cats = defaultdict(list)
    late_cats = defaultdict(list)

    for ep in train_eps:
        for t in ep["transcript"]:
            if t["role"] != ep["role"]:
                continue
            cats = _count_categories(t["text"])
            bucket = early_cats if ep["step"] <= 1 else (late_cats if ep["step"] >= 5 else None)
            if bucket is not None:
                for k, v in cats.items():
                    bucket[k].append(v)

    # Two-panel figure: left = word-level (substantive vs scaffolding),
    # right = phrase counts (backtracks and meta-commentary)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)

    # Left panel: word counts (substantive, scaffolding)
    word_cats = ["substantive", "scaffolding"]
    word_labels = ["Substantive\nargument", "Structural\nscaffolding"]
    word_colors = [COLORS["blue"], COLORS["orange"]]

    early_word_means = [np.mean(early_cats[c]) for c in word_cats]
    late_word_means = [np.mean(late_cats[c]) for c in word_cats]

    x1 = np.arange(len(word_cats))
    width = 0.35
    bars1a = ax1.bar(x1 - width / 2, early_word_means, width, label="Early (steps 0-1)",
                     color=[c + "99" for c in word_colors], edgecolor=word_colors, linewidth=1.2)
    bars1b = ax1.bar(x1 + width / 2, late_word_means, width, label="Late (steps 5+)",
                     color=word_colors, edgecolor=word_colors, linewidth=1.2)

    for bars in [bars1a, bars1b]:
        for bar in bars:
            h = bar.get_height()
            if h > 10:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:.0f}",
                         ha="center", va="bottom", fontsize=8, color=COLORS["dark"])

    ax1.set_title("Words per Turn")
    ax1.set_ylabel("Mean words")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(word_labels)
    ax1.legend(loc="upper right", fontsize=9)

    # Right panel: phrase counts (backtrack, meta)
    phrase_cats = ["backtrack_phrases", "meta_phrases"]
    phrase_labels = ["Backtrack\nphrases", "Meta-commentary\nphrases"]
    phrase_colors = [COLORS["red"], COLORS["purple"]]

    early_phrase_means = [np.mean(early_cats[c]) for c in phrase_cats]
    late_phrase_means = [np.mean(late_cats[c]) for c in phrase_cats]

    x2 = np.arange(len(phrase_cats))
    bars2a = ax2.bar(x2 - width / 2, early_phrase_means, width, label="Early (steps 0-1)",
                     color=[c + "99" for c in phrase_colors], edgecolor=phrase_colors, linewidth=1.2)
    bars2b = ax2.bar(x2 + width / 2, late_phrase_means, width, label="Late (steps 5+)",
                     color=phrase_colors, edgecolor=phrase_colors, linewidth=1.2)

    for bars in [bars2a, bars2b]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.3:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}",
                         ha="center", va="bottom", fontsize=8, color=COLORS["dark"])

    ax2.set_title("Phrase Counts per Turn")
    ax2.set_ylabel("Mean count")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(phrase_labels)
    ax2.legend(loc="upper right", fontsize=9)

    # Annotate backtrack growth
    bt_early = early_phrase_means[0]
    bt_late = late_phrase_means[0]
    if bt_early > 0:
        bt_growth = (bt_late / bt_early - 1) * 100
        if bt_growth > 50:
            annotate_insight(
                ax2, f"+{bt_growth:.0f}%",
                xy=(0 + width / 2, bt_late),
                xytext=(0.8, bt_late + 2),
                color=COLORS["red"],
            )

    fig.suptitle(
        "Training inflates backtrack loops and structural scaffolding",
        fontsize=13, fontweight="bold", color=COLORS["dark"], y=1.02,
    )

    save(fig, "fig_protocol_verbosity_anatomy")
    plt.close(fig)


# ── Figure 4: Accuracy vs Length Scatter ─────────────────────────────────────


def fig_accuracy_vs_length():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect all runs, all episodes
    all_points = []
    for run in RUNS:
        episodes = load_episodes(run)
        train_eps = [e for e in episodes if e["split"] == "train"]

        for ep in train_eps:
            wc = sum(len(t["text"].split()) for t in ep["transcript"] if t["role"] == ep["role"])
            acc_key = f"accuracy.{ep['role']}"
            acc = ep["signals"].get(acc_key, None)
            if acc is not None:
                all_points.append({
                    "run": run,
                    "step": ep["step"],
                    "word_count": wc,
                    "accuracy": acc,
                })

    wcs = np.array([p["word_count"] for p in all_points])
    accs = np.array([p["accuracy"] for p in all_points])

    # Bin by word count — extend to cover all data
    max_wc = int(np.ceil(wcs.max() / 500) * 500) + 500
    wc_bins = np.arange(0, max_wc + 1, 500)
    bin_centers = (wc_bins[:-1] + wc_bins[1:]) / 2
    bin_accs = []
    bin_counts = []
    bin_cis = []  # 95% binomial CI half-width
    for i in range(len(wc_bins) - 1):
        mask = (wcs >= wc_bins[i]) & (wcs < wc_bins[i + 1])
        n = mask.sum()
        if n > 0:
            p_hat = accs[mask].mean()
            bin_accs.append(p_hat)
            bin_counts.append(n)
            # Wilson score interval approximation
            ci = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n) if n > 1 else 0.5
            bin_cis.append(ci)
        else:
            bin_accs.append(np.nan)
            bin_counts.append(0)
            bin_cis.append(0)

    bin_accs = np.array(bin_accs)
    bin_cis = np.array(bin_cis)

    # Only plot bins with enough data
    valid = np.array([i for i in range(len(bin_accs)) if not np.isnan(bin_accs[i]) and bin_counts[i] >= 10])

    # Main visualization: binned proportion with CI
    ax.fill_between(
        bin_centers[valid],
        (bin_accs[valid] - bin_cis[valid]).clip(0, 1),
        (bin_accs[valid] + bin_cis[valid]).clip(0, 1),
        alpha=0.2, color=COLORS["blue"],
    )
    ax.plot(
        bin_centers[valid], bin_accs[valid],
        "o-", color=COLORS["blue"], markersize=6, linewidth=2.0,
        label="Binned accuracy (95% CI)", zorder=5,
    )

    # Add sample counts as secondary annotation
    for i in valid:
        if bin_counts[i] >= 10:
            ax.text(
                bin_centers[i], bin_accs[i] - bin_cis[i] - 0.04,
                f"n={bin_counts[i]}", ha="center", va="top",
                fontsize=6.5, color=COLORS["light"],
            )

    # Point-biserial correlation
    corr = np.corrcoef(wcs, accs)[0, 1]

    ax.set_title("Accuracy vs Response Length (All Protocols)")
    ax.set_xlabel("Total words per episode (this debater)")
    ax.set_ylabel("Proportion correct")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=9)

    ax.text(
        0.97, 0.03, f"r_pb = {corr:.3f}  (n={len(all_points):,})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=COLORS["dark"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"),
    )

    fig.suptitle(
        "Longer responses correlate with lower accuracy",
        fontsize=13, fontweight="bold", color=COLORS["dark"], y=1.02,
    )

    save(fig, "fig_accuracy_vs_length")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Generating verbosity and quality figures...")
    fig_token_growth()
    fig_parse_success()
    fig_verbosity_anatomy()
    fig_accuracy_vs_length()
    print("Done.")
