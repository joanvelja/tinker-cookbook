"""Generate evaluation metric figures from debate training logs."""

import json
import os
import glob
import zipfile

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

# Colorblind-safe palette (Wong 2011)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERMILLION = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN = "#56B4E9"
C_YELLOW = "#F0E442"

OUTDIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(os.path.dirname(OUTDIR), os.pardir, "logs", "thinking-experiment")
RUNG1 = os.path.join(LOGDIR, "rung1-no-think", "metrics.jsonl")
RUNG2 = os.path.join(LOGDIR, "rung2-private-think", "metrics.jsonl")


# ── Data loading ───────────────────────────────────────────────────────────
def load_metrics(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_key(d, key, default=None):
    return d.get(key, default)


rung1 = load_metrics(RUNG1)
rung2 = load_metrics(RUNG2)


def extract_series(data, key_prefix, metric):
    """Extract train and test series for a metric."""
    train_key = f"{key_prefix}/{metric}"
    test_key = f"test/{key_prefix}/{metric}"
    steps, train_vals, test_steps, test_vals = [], [], [], []
    for d in data:
        step = d["step"]
        tv = d.get(train_key)
        if tv is not None:
            steps.append(step)
            train_vals.append(tv)
        ttv = d.get(test_key)
        if ttv is not None:
            test_steps.append(step)
            test_vals.append(ttv)
    return steps, train_vals, test_steps, test_vals


def mean_accuracy(data, prefix="env/all"):
    """Compute mean of debater_a and debater_b accuracy."""
    steps, vals = [], []
    for d in data:
        step = d["step"]
        a = d.get(f"{prefix}/accuracy.debater_a")
        b = d.get(f"{prefix}/accuracy.debater_b")
        if a is not None and b is not None:
            steps.append(step)
            vals.append((a + b) / 2)
    return steps, vals


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Test-set accuracy over training checkpoints
# ═══════════════════════════════════════════════════════════════════════════
def fig_eval_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Rung 1 — no think
    r1_train_steps, r1_train_acc = mean_accuracy(rung1, "env/all")
    r1_test_steps, r1_test_acc = mean_accuracy(rung1, "test/env/all")

    # Rung 2 — private think
    r2_train_steps, r2_train_acc = mean_accuracy(rung2, "env/all")
    r2_test_steps, r2_test_acc = mean_accuracy(rung2, "test/env/all")

    # Train lines (lighter, thinner)
    ax.plot(
        r1_train_steps, r1_train_acc,
        color=C_BLUE, alpha=0.35, linewidth=1.2, linestyle="--",
        label="No-think train",
    )
    ax.plot(
        r2_train_steps, r2_train_acc,
        color=C_ORANGE, alpha=0.35, linewidth=1.2, linestyle="--",
        label="Private-think train",
    )

    # Test lines (bold, with markers)
    ax.plot(
        r1_test_steps, r1_test_acc,
        color=C_BLUE, linewidth=2.2, marker="o", markersize=8,
        label="No-think test", zorder=5,
    )
    ax.plot(
        r2_test_steps, r2_test_acc,
        color=C_ORANGE, linewidth=2.2, marker="s", markersize=8,
        label="Private-think test", zorder=5,
    )

    # Annotate test values — stagger to avoid overlap
    r1_offsets = [(-20, -14), (-18, 10), (0, 10)]  # (x, y) offset per point
    for (s, v), (xo, yo) in zip(zip(r1_test_steps, r1_test_acc), r1_offsets):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points",
                    xytext=(xo, yo), ha="center", fontsize=9, color=C_BLUE)
    r2_offsets = [(18, 10), (18, -14)]
    for (s, v), (xo, yo) in zip(zip(r2_test_steps, r2_test_acc), r2_offsets):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points",
                    xytext=(xo, yo), ha="center", fontsize=9, color=C_ORANGE)

    # Chance line
    ax.axhline(0.25, color="grey", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(max(r1_train_steps) * 0.95, 0.255, "chance (4-choice)", ha="right",
            fontsize=9, color="grey", alpha=0.7)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean accuracy (debater A + B) / 2")
    ax.set_title("Test accuracy degrades alongside train accuracy")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0.15, 0.55)
    ax.set_xlim(-0.5, max(r1_train_steps) + 0.5)

    fig.savefig(os.path.join(OUTDIR, "fig_eval_accuracy.png"))
    plt.close(fig)
    print("Saved fig_eval_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Test-train divergence for key metrics
# ═══════════════════════════════════════════════════════════════════════════
def fig_eval_vs_train():
    metrics = {
        "accuracy_mean": lambda d, pfx: (
            (d.get(f"{pfx}/accuracy.debater_a", 0) + d.get(f"{pfx}/accuracy.debater_b", 0)) / 2
            if d.get(f"{pfx}/accuracy.debater_a") is not None else None
        ),
        "disagreement": lambda d, pfx: d.get(f"{pfx}/disagreement"),
        "draw_rate": lambda d, pfx: d.get(f"{pfx}/draw_rate"),
        "truth_surfaced": lambda d, pfx: d.get(f"{pfx}/truth_surfaced"),
    }

    # Rung1 eval steps: 0, 5, 10
    eval_indices = [0, 5, 10]
    eval_steps = [rung1[i]["step"] for i in eval_indices]

    gaps = {m: [] for m in metrics}
    for idx in eval_indices:
        d = rung1[idx]
        for mname, extractor in metrics.items():
            test_val = extractor(d, "test/env/all")
            train_val = extractor(d, "env/all")
            if test_val is not None and train_val is not None:
                gaps[mname].append(test_val - train_val)
            else:
                gaps[mname].append(0)

    metric_labels = list(metrics.keys())
    n_metrics = len(metric_labels)
    n_steps = len(eval_steps)
    x = np.arange(n_metrics)
    width = 0.22

    colors = [C_CYAN, C_BLUE, C_VERMILLION]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (step, color) in enumerate(zip(eval_steps, colors)):
        offsets = [gaps[m][i] for m in metric_labels]
        bars = ax.bar(x + (i - 1) * width, offsets, width, label=f"Step {step}",
                      color=color, edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, offsets):
            if val >= 0:
                ypos = bar.get_height()
                va = "bottom"
                pad = 0.003
            else:
                ypos = bar.get_y()
                va = "top"
                pad = -0.003
            # Rotate labels on small values to avoid overlap
            rot = 30 if abs(val) < 0.02 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, ypos + pad,
                    f"{val:+.2f}", ha="center", va=va, fontsize=10,
                    rotation=rot)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_labels])
    ax.set_ylabel("Test - Train gap")
    ax.set_title("Test set retains accuracy better than train\n(positive = test > train)")
    ax.legend(title="Eval checkpoint", loc="lower left")

    fig.savefig(os.path.join(OUTDIR, "fig_eval_vs_train.png"))
    plt.close(fig)
    print("Saved fig_eval_vs_train.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Seat bias on test set vs train set
# ═══════════════════════════════════════════════════════════════════════════
def fig_eval_seat_bias():
    fig, ax = plt.subplots(figsize=(7, 5))

    def seat_bias_series(data, prefix):
        steps, vals = [], []
        for d in data:
            wa = d.get(f"{prefix}/win_rate.debater_a")
            wb = d.get(f"{prefix}/win_rate.debater_b")
            if wa is not None and wb is not None:
                steps.append(d["step"])
                vals.append(wb - wa)
        return steps, vals

    # Rung1
    r1_train_s, r1_train_v = seat_bias_series(rung1, "env/all")
    r1_test_s, r1_test_v = seat_bias_series(rung1, "test/env/all")

    # Rung2
    r2_train_s, r2_train_v = seat_bias_series(rung2, "env/all")
    r2_test_s, r2_test_v = seat_bias_series(rung2, "test/env/all")

    ax.plot(r1_train_s, r1_train_v, color=C_BLUE, alpha=0.4, linestyle="--",
            linewidth=1.2, label="No-think train")
    ax.plot(r1_test_s, r1_test_v, color=C_BLUE, linewidth=2.2, marker="o",
            markersize=7, label="No-think test", zorder=5)

    ax.plot(r2_train_s, r2_train_v, color=C_ORANGE, alpha=0.4, linestyle="--",
            linewidth=1.2, label="Private-think train")
    ax.plot(r2_test_s, r2_test_v, color=C_ORANGE, linewidth=2.2, marker="s",
            markersize=7, label="Private-think test", zorder=5)

    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.text(0.5, 0.02, "no seat bias", fontsize=9, color="grey", alpha=0.7)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Seat bias (win_rate B - win_rate A)")
    ax.set_title("Seat B advantage persists on held-out data")
    ax.legend(loc="best", framealpha=0.9)

    fig.savefig(os.path.join(OUTDIR, "fig_eval_seat_bias.png"))
    plt.close(fig)
    print("Saved fig_eval_seat_bias.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Trained vs opponent from .eval files
# ═══════════════════════════════════════════════════════════════════════════
def fig_eval_id_metrics():
    eval_dir = os.path.expanduser("~/inspect-logs")
    eval_files = sorted(glob.glob(os.path.join(eval_dir, "*.eval")), key=os.path.getmtime)

    id_metrics = [
        "id/win_rate.trained",
        "id/accuracy.trained",
        "id/accuracy.opponent",
        "id/wrong_and_wins.trained",
    ]
    pretty_labels = [
        "Win rate\n(trained)",
        "Accuracy\n(trained)",
        "Accuracy\n(opponent)",
        "Wrong & wins\n(trained)",
    ]

    # Group by prompts_ref
    groups = {}  # prompts_ref -> list of {metric: mean}
    for ef in eval_files[-8:]:
        with zipfile.ZipFile(ef) as zf:
            names = zf.namelist()
            header = None
            for c in ["header.json", "_journal/start.json"]:
                if c in names:
                    with zf.open(c) as fh:
                        header = json.load(fh)
                    break
            if not header:
                continue

            task_args = header.get("eval", {}).get("task_args", {})
            prompts_ref = task_args.get("prompts_ref", "unknown")

            sample_files = [n for n in names if n.startswith("samples/")]
            if not sample_files:
                continue

            agg = {k: [] for k in id_metrics}
            for sf_name in sample_files:
                with zf.open(sf_name) as sf:
                    sample = json.load(sf)
                value = sample.get("scores", {}).get("_scorer", {}).get("value", {})
                for k in id_metrics:
                    v = value.get(k)
                    if v is not None:
                        agg[k].append(v)

            means = {}
            for k in id_metrics:
                if agg[k]:
                    means[k] = np.mean(agg[k])
            if means:
                groups.setdefault(prompts_ref, []).append(means)

    # Compute per-group averages across eval runs
    group_labels = sorted(groups.keys())
    group_avgs = {}
    group_stds = {}
    for g in group_labels:
        runs = groups[g]
        avg = {}
        std = {}
        for k in id_metrics:
            vals = [r[k] for r in runs if k in r]
            avg[k] = np.mean(vals) if vals else 0
            std[k] = np.std(vals) if len(vals) > 1 else 0
        group_avgs[g] = avg
        group_stds[g] = std

    n_metrics = len(id_metrics)
    n_groups = len(group_labels)
    x = np.arange(n_metrics)
    width = 0.35

    colors = {
        "open_selfplay": C_BLUE,
        "open_selfplay_private": C_ORANGE,
    }
    display_names = {
        "open_selfplay": "No-think",
        "open_selfplay_private": "Private-think",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(group_labels):
        vals = [group_avgs[g].get(k, 0) for k in id_metrics]
        errs = [group_stds[g].get(k, 0) for k in id_metrics]
        color = colors.get(g, C_GREEN)
        label = display_names.get(g, g)
        bars = ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs,
                      label=label, color=color, edgecolor="white",
                      linewidth=0.5, capsize=3, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Reference line at 0.5 for win rate
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels)
    ax.set_ylabel("Mean across eval runs")
    ax.set_title("Trained model wins ~40% with accuracy comparable to opponent")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 0.65)

    fig.savefig(os.path.join(OUTDIR, "fig_eval_id_metrics.png"))
    plt.close(fig)
    print("Saved fig_eval_id_metrics.png")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_eval_accuracy()
    fig_eval_vs_train()
    fig_eval_seat_bias()
    fig_eval_id_metrics()
    print("\nAll figures generated.")
