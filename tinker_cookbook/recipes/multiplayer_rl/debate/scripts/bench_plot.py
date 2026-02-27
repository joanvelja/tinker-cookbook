"""Publication-quality plots from bench_latency.py JSONL output.

Reads three row types (call, run_summary, sweep_call) and produces 10 plots
covering Gantt timelines, scaling laws, speedup analysis, and diagnostics.

Usage:
    uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.bench_plot \
        results.jsonl --output-dir plots/ --format pdf --dpi 300 --dark-theme
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

Row = dict[str, Any]


def load_jsonl(path: Path) -> list[Row]:
    rows: list[Row] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[Row]) -> tuple[list[Row], list[Row], list[Row]]:
    """Split into (calls, run_summaries, sweep_calls)."""
    calls = [r for r in rows if r.get("type") == "call"]
    summaries = [r for r in rows if r.get("type") == "run_summary"]
    sweeps = [r for r in rows if r.get("type") == "sweep_call"]
    return calls, summaries, sweeps


# ---------------------------------------------------------------------------
# Fitting functions (Laws 1-3)
# ---------------------------------------------------------------------------


def law1_prefill(n: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    """t = c + a*n + b*n^2"""
    return c + a * n + b * n**2


def law2_reservation(x: np.ndarray, c: float, d: float, r: float) -> np.ndarray:
    """t = c + d*output_tokens + r*max_tokens. x = (output_tokens, max_tokens) stacked."""
    output_tokens, max_tokens = x
    return c + d * output_tokens + r * max_tokens


def law3_usl(n: np.ndarray, sigma: float, kappa: float) -> np.ndarray:
    """USL: throughput(N) = N / (1 + sigma*(N-1) + kappa*N*(N-1))"""
    return n / (1 + sigma * (n - 1) + kappa * n * (n - 1))


def _safe_fit(func, xdata, ydata, p0=None, maxfev=5000):
    """Curve fit with fallback to None on failure."""
    try:
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0, maxfev=maxfev)
        return popt, pcov
    except (RuntimeError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Statistics: paired permutation test
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.10,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute log-ratio speedup with paired bootstrap CI.

    Returns (point_estimate, ci_lo, ci_hi) as speedup ratios.
    CI always contains the point estimate.
    """
    assert len(baseline) == len(treatment)
    n = len(baseline)
    if n == 0:
        return 1.0, 1.0, 1.0

    log_ratios = np.log(baseline) - np.log(treatment)
    point = np.exp(np.mean(log_ratios))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(log_ratios[idx])

    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return point, np.exp(lo), np.exp(hi)


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

# Arm colors — deterministic palette.
ARM_COLORS = {
    "A0": "#4C72B0",
    "A1": "#DD8452",
    "A2": "#55A868",
    "A3": "#C44E52",
}
ACTOR_COLORS = {
    "trained": "#4C72B0",
    "opponent": "#DD8452",
    "judge": "#55A868",
    "debater_a": "#4C72B0",
    "debater_b": "#DD8452",
}


def _get_actor(row: Row) -> str:
    """Get actor/role label from a call row, tolerating both field names."""
    return row.get("actor", row.get("role", "?"))


def _get_wall_s(row: Row) -> float:
    """Get wall-clock seconds from a row, tolerating both field names."""
    return row.get("wall_s", row.get("time/step_wall_s", 0.0))


def _get_block(row: Row) -> int:
    """Get block index from a row, tolerating both field names."""
    return row.get("block", row.get("block_id", 0))


def _get_run_wall_s(row: Row) -> float:
    """Get run wall-clock seconds from a summary row."""
    return row.get("wall_s", row.get("wall_clock_s", 0.0))


def setup_style(dark: bool) -> None:
    if dark:
        plt.style.use("dark_background")
        sns.set_palette("bright")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("deep")
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


# ---------------------------------------------------------------------------
# Plot 1: Gantt timeline
# ---------------------------------------------------------------------------


def plot_gantt(calls: list[Row], ax: plt.Axes) -> None:
    """Swimlanes by actor x env x problem."""
    # Gantt requires t_submit for positioning; skip if absent.
    timed_calls = [c for c in calls if "t_submit" in c]
    if not timed_calls:
        ax.text(0.5, 0.5, "No call data with t_submit", transform=ax.transAxes, ha="center")
        return

    # Determine global t0.
    t0 = min(c["t_submit"] for c in timed_calls)

    # Build lane labels: actor/problem/env.
    lanes: dict[str, int] = {}
    for c in timed_calls:
        actor = _get_actor(c)
        p_idx = c.get("problem_index", c.get("problem_id", 0))
        e_idx = c.get("env_index", c.get("env_id", 0))
        key = f"{actor}/p{p_idx}/e{e_idx}"
        if key not in lanes:
            lanes[key] = len(lanes)

    for c in timed_calls:
        actor = _get_actor(c)
        p_idx = c.get("problem_index", c.get("problem_id", 0))
        e_idx = c.get("env_index", c.get("env_id", 0))
        key = f"{actor}/p{p_idx}/e{e_idx}"
        y = lanes[key]
        start = c["t_submit"] - t0
        dur = _get_wall_s(c)
        color = ACTOR_COLORS.get(actor, "#888888")
        ax.barh(y, dur, left=start, height=0.7, color=color, alpha=0.8, edgecolor="none")

    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(list(lanes.keys()), fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title("Gantt Timeline")
    ax.invert_yaxis()

    # Legend.
    seen_actors = {_get_actor(c) for c in timed_calls}
    handles = [
        mpatches.Patch(color=v, label=k) for k, v in ACTOR_COLORS.items() if k in seen_actors
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)


# ---------------------------------------------------------------------------
# Plot 2: Prefill scaling curve (Law 1)
# ---------------------------------------------------------------------------


def plot_prefill_scaling(calls: list[Row], sweeps: list[Row], ax: plt.Axes) -> None:
    """Latency vs input_tokens with quadratic fit."""
    # Prefer sweep data (S2) if available.
    points = [s for s in sweeps if s.get("sweep", "").startswith("S2")]
    if not points:
        # Fall back to call rows that have input_tokens.
        points = [c for c in calls if "input_tokens" in c]

    if not points:
        ax.text(0.5, 0.5, "No data with input_tokens", transform=ax.transAxes, ha="center")
        return

    x = np.array([p["input_tokens"] for p in points], dtype=float)
    y = np.array([_get_wall_s(p) for p in points], dtype=float)

    ax.scatter(x, y, s=20, alpha=0.6, zorder=3)

    # Fit Law 1.
    popt, _ = _safe_fit(law1_prefill, x, y, p0=[1.0, 1e-4, 1e-9])
    if popt is not None:
        x_fit = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            x_fit,
            law1_prefill(x_fit, *popt),
            "r-",
            lw=1.5,
            label=f"c={popt[0]:.2f}, a={popt[1]:.2e}, b={popt[2]:.2e}",
        )

        # Local elasticity annotation at median.
        x_med = np.median(x)
        deriv = popt[1] + 2 * popt[2] * x_med
        elasticity = deriv * x_med / law1_prefill(np.array([x_med]), *popt)[0]
        ax.annotate(
            f"elasticity={elasticity:.2f}",
            xy=(x_med, law1_prefill(np.array([x_med]), *popt)[0]),
            fontsize=8,
            color="red",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Prefill Scaling (Law 1)")


# ---------------------------------------------------------------------------
# Plot 3: Reservation tax (Law 2)
# ---------------------------------------------------------------------------


def plot_reservation_tax(calls: list[Row], sweeps: list[Row], ax: plt.Axes) -> None:
    """Efficiency vs waste ratio."""
    # Prefer sweep data (S1) if available.
    points = [s for s in sweeps if s.get("sweep", "").startswith("S1")]
    if not points:
        # Fall back to call rows that have both max_tokens and output_tokens.
        points = [c for c in calls if "max_tokens" in c and c.get("output_tokens", 0) > 0]

    if not points:
        ax.text(0.5, 0.5, "No data with max_tokens", transform=ax.transAxes, ha="center")
        return

    waste = np.array([p["max_tokens"] / max(p["output_tokens"], 1) for p in points])
    efficiency = np.array([p["output_tokens"] / max(_get_wall_s(p), 0.001) for p in points])

    ax.scatter(waste, efficiency, s=20, alpha=0.6)
    ax.set_xlabel("Waste ratio (max_tokens / actual_output)")
    ax.set_ylabel("Efficiency (tok/s)")
    ax.set_title("Reservation Tax (Law 2)")


# ---------------------------------------------------------------------------
# Plot 4: Concurrency saturation (Law 3)
# ---------------------------------------------------------------------------


def plot_concurrency_saturation(sweeps: list[Row], ax: plt.Axes) -> None:
    """Throughput vs N with USL fit."""
    points = [s for s in sweeps if s.get("sweep", "").startswith("S3")]
    if not points:
        ax.text(0.5, 0.5, "No sweep S3 data", transform=ax.transAxes, ha="center")
        return

    # Group by concurrency level.
    by_n: dict[int, list[float]] = {}
    for p in points:
        n = p["level"]
        throughput = p.get("output_tokens", 1) / max(p["wall_s"], 0.001)
        by_n.setdefault(n, []).append(throughput)

    ns = sorted(by_n.keys())
    means = np.array([np.mean(by_n[n]) for n in ns])
    # Normalize so N=1 throughput ~ 1 for USL fitting.
    baseline = means[0] if means[0] > 0 else 1.0
    norm_means = means / baseline
    n_arr = np.array(ns, dtype=float)

    ax.scatter(n_arr, norm_means, s=40, zorder=3)

    popt, _ = _safe_fit(law3_usl, n_arr, norm_means, p0=[0.01, 0.001])
    if popt is not None:
        n_fit = np.linspace(1, n_arr.max(), 200)
        ax.plot(
            n_fit, law3_usl(n_fit, *popt), "r-", lw=1.5, label=f"σ={popt[0]:.3f}, κ={popt[1]:.4f}"
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Concurrency (N)")
    ax.set_ylabel("Normalized throughput")
    ax.set_title("Concurrency Saturation (Law 3 / USL)")


# ---------------------------------------------------------------------------
# Plot 5: Critical-path waterfall (Law 4)
# ---------------------------------------------------------------------------


def plot_critical_path_waterfall(calls: list[Row], summaries: list[Row], ax: plt.Axes) -> None:
    """Predicted vs actual rollout time per run, with residuals."""
    if not summaries:
        ax.text(0.5, 0.5, "No run_summary data", transform=ax.transAxes, ha="center")
        return

    # Group calls by run_id.
    calls_by_run: dict[str, list[Row]] = {}
    for c in calls:
        calls_by_run.setdefault(c["run_id"], []).append(c)

    run_ids = []
    actual_times = []
    predicted_times = []

    for s in summaries:
        rid = s["run_id"]
        actual = _get_run_wall_s(s)
        run_calls = calls_by_run.get(rid, [])
        if not run_calls:
            continue
        # Predicted = sum of individual call latencies (sequential critical path).
        predicted = sum(_get_wall_s(c) for c in run_calls)
        run_ids.append(rid)
        actual_times.append(actual)
        predicted_times.append(predicted)

    if not run_ids:
        ax.text(0.5, 0.5, "No matched runs", transform=ax.transAxes, ha="center")
        return

    x = np.arange(len(run_ids))
    width = 0.35
    ax.bar(x - width / 2, predicted_times, width, label="Predicted (sum of calls)", alpha=0.8)
    ax.bar(x + width / 2, actual_times, width, label="Actual", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Time (s)")
    ax.set_title("Critical-Path Waterfall (Law 4)")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Plot 6: Speedup bars
# ---------------------------------------------------------------------------


def plot_speedup_bars(summaries: list[Row], ax: plt.Axes) -> None:
    """A0=1.0 baseline, other arms with 90% CIs from paired permutation."""
    if not summaries:
        ax.text(0.5, 0.5, "No run_summary data", transform=ax.transAxes, ha="center")
        return

    # Group by arm and block.
    by_arm_block: dict[str, dict[int, float]] = {}
    for s in summaries:
        arm = s["arm"]
        block = _get_block(s)
        by_arm_block.setdefault(arm, {})[block] = _get_run_wall_s(s)

    if "A0" not in by_arm_block:
        ax.text(0.5, 0.5, "No A0 baseline", transform=ax.transAxes, ha="center")
        return

    baseline_blocks = by_arm_block["A0"]
    arms = sorted(by_arm_block.keys())

    bar_data: list[tuple[str, float, float, float]] = []
    for arm in arms:
        arm_blocks = by_arm_block[arm]
        # Pair by block.
        common_blocks = sorted(set(baseline_blocks) & set(arm_blocks))
        if not common_blocks:
            continue
        b = np.array([baseline_blocks[bid] for bid in common_blocks])
        t = np.array([arm_blocks[bid] for bid in common_blocks])
        point, ci_lo, ci_hi = paired_bootstrap_ci(b, t)
        bar_data.append((arm, point, ci_lo, ci_hi))

    if not bar_data:
        ax.text(0.5, 0.5, "No paired blocks", transform=ax.transAxes, ha="center")
        return

    labels = [d[0] for d in bar_data]
    points = [d[1] for d in bar_data]
    lo_err = [d[1] - d[2] for d in bar_data]
    hi_err = [d[3] - d[1] for d in bar_data]
    colors = [ARM_COLORS.get(label, "#888888") for label in labels]

    x = np.arange(len(labels))
    ax.bar(
        x,
        points,
        yerr=[lo_err, hi_err],
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup (vs A0)")
    ax.set_title("Speedup by Arm (90% CI)")


# ---------------------------------------------------------------------------
# Plot 7: Paired block spaghetti
# ---------------------------------------------------------------------------


def plot_paired_spaghetti(summaries: list[Row], ax: plt.Axes) -> None:
    """Same block connected across configs."""
    if not summaries:
        ax.text(0.5, 0.5, "No run_summary data", transform=ax.transAxes, ha="center")
        return

    by_arm: dict[str, dict[int, float]] = {}
    for s in summaries:
        by_arm.setdefault(s["arm"], {})[_get_block(s)] = _get_run_wall_s(s)

    arms = sorted(by_arm.keys())
    if len(arms) < 2:
        ax.text(0.5, 0.5, "Need 2+ arms", transform=ax.transAxes, ha="center")
        return

    all_blocks = sorted({b for arm_data in by_arm.values() for b in arm_data})
    x = np.arange(len(arms))

    for block_id in all_blocks:
        ys = []
        valid_x = []
        for i, arm in enumerate(arms):
            if block_id in by_arm[arm]:
                ys.append(by_arm[arm][block_id])
                valid_x.append(i)
        if len(valid_x) > 1:
            ax.plot(
                valid_x,
                ys,
                "o-",
                alpha=0.5,
                markersize=4,
                label=f"block {block_id}" if block_id == all_blocks[0] else None,
            )

    # Mean line.
    means = []
    for arm in arms:
        vals = list(by_arm[arm].values())
        means.append(np.mean(vals) if vals else 0)
    ax.plot(x, means, "s-", color="black", markersize=6, linewidth=2, label="mean", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("Wall clock (s)")
    ax.set_title("Paired Block Spaghetti")
    ax.legend(fontsize=7, loc="best")


# ---------------------------------------------------------------------------
# Plot 8: Latency distribution (violin + strip)
# ---------------------------------------------------------------------------


def plot_latency_distribution(calls: list[Row], ax: plt.Axes) -> None:
    """Violin + strip by actor/config."""
    if not calls:
        ax.text(0.5, 0.5, "No call data", transform=ax.transAxes, ha="center")
        return

    # Build a flat structure for seaborn.
    actors = []
    latencies = []
    arms = []
    for c in calls:
        actors.append(_get_actor(c))
        latencies.append(_get_wall_s(c))
        arms.append(c.get("arm", "?"))

    import pandas as pd

    df = pd.DataFrame({"actor": actors, "latency_s": latencies, "arm": arms})
    df["group"] = df["arm"] + " / " + df["actor"]

    groups = sorted(df["group"].unique())
    if len(groups) > 20:
        # Too many groups — collapse to actor only.
        df["group"] = df["actor"]
        groups = sorted(df["group"].unique())

    sns.violinplot(data=df, x="group", y="latency_s", ax=ax, inner=None, alpha=0.3, cut=0)
    sns.stripplot(data=df, x="group", y="latency_s", ax=ax, size=3, alpha=0.5, jitter=True)

    ax.tick_params(axis="x", rotation=45, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Distribution")


# ---------------------------------------------------------------------------
# Plot 9: Scatter prompt tokens vs latency with Law 1 overlay
# ---------------------------------------------------------------------------


def plot_prompt_vs_latency(calls: list[Row], ax: plt.Axes) -> None:
    """Scatter by actor/config with Law 1 fit overlaid."""
    # Only include calls that have input_tokens.
    valid = [c for c in calls if "input_tokens" in c]
    if not valid:
        ax.text(0.5, 0.5, "No call data with input_tokens", transform=ax.transAxes, ha="center")
        return

    actors = sorted({_get_actor(c) for c in valid})
    all_x = []
    all_y = []
    for actor in actors:
        pts = [c for c in valid if _get_actor(c) == actor]
        x = [p["input_tokens"] for p in pts]
        y = [_get_wall_s(p) for p in pts]
        all_x.extend(x)
        all_y.extend(y)
        color = ACTOR_COLORS.get(actor, "#888888")
        ax.scatter(x, y, s=15, alpha=0.5, label=actor, color=color)

    # Global Law 1 fit.
    if all_x:
        xa = np.array(all_x, dtype=float)
        ya = np.array(all_y, dtype=float)
        popt, _ = _safe_fit(law1_prefill, xa, ya, p0=[1.0, 1e-4, 1e-9])
        if popt is not None:
            x_fit = np.linspace(xa.min(), xa.max(), 200)
            ax.plot(x_fit, law1_prefill(x_fit, *popt), "k--", lw=1.5, alpha=0.7, label="Law 1 fit")

    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Prompt Tokens vs Latency")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Plot 10: Incident overlay (retries/backoff)
# ---------------------------------------------------------------------------


def plot_incidents(calls: list[Row], ax: plt.Axes) -> None:
    """Retries/backoff events overlaid on latency timeline."""
    # Incidents require t_submit for timeline positioning.
    timed_calls = [c for c in calls if "t_submit" in c]
    if not timed_calls:
        ax.text(0.5, 0.5, "No call data with t_submit", transform=ax.transAxes, ha="center")
        return

    t0 = min(c["t_submit"] for c in timed_calls)
    times = np.array([c["t_submit"] - t0 for c in timed_calls])
    latencies = np.array([_get_wall_s(c) for c in timed_calls])

    ax.scatter(times, latencies, s=10, alpha=0.3, color="#4C72B0", label="calls")

    # Overlay incidents.
    incident_times = []
    incident_latencies = []
    incident_labels = []
    for c in timed_calls:
        if c.get("retries", 0) > 0 or c.get("had_429", False) or c.get("backoff_s", 0) > 0:
            incident_times.append(c["t_submit"] - t0)
            incident_latencies.append(_get_wall_s(c))
            parts = []
            if c.get("retries", 0) > 0:
                parts.append(f"retry={c['retries']}")
            if c.get("had_429", False):
                parts.append("429")
            if c.get("backoff_s", 0) > 0:
                parts.append(f"backoff={c['backoff_s']:.1f}s")
            incident_labels.append(", ".join(parts))

    if incident_times:
        ax.scatter(
            incident_times,
            incident_latencies,
            s=60,
            marker="x",
            color="red",
            linewidths=2,
            zorder=5,
            label=f"incidents ({len(incident_times)})",
        )
        # Annotate first few.
        for i, (t, lat, lbl) in enumerate(zip(incident_times, incident_latencies, incident_labels)):
            if i < 5:
                ax.annotate(
                    lbl,
                    (t, lat),
                    fontsize=6,
                    color="red",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Incident Overlay")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def make_all_plots(
    rows: list[Row],
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> list[Path]:
    """Generate all 10 plots, return list of saved paths."""
    calls, summaries, sweeps = split_rows(rows)

    plots = [
        ("01_gantt", plot_gantt, (calls,)),
        ("02_prefill_scaling", plot_prefill_scaling, (calls, sweeps)),
        ("03_reservation_tax", plot_reservation_tax, (calls, sweeps)),
        ("04_concurrency_saturation", plot_concurrency_saturation, (sweeps,)),
        ("05_critical_path", plot_critical_path_waterfall, (calls, summaries)),
        ("06_speedup_bars", plot_speedup_bars, (summaries,)),
        ("07_paired_spaghetti", plot_paired_spaghetti, (summaries,)),
        ("08_latency_distribution", plot_latency_distribution, (calls,)),
        ("09_prompt_vs_latency", plot_prompt_vs_latency, (calls,)),
        ("10_incidents", plot_incidents, (calls,)),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for name, plot_fn, args in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            plot_fn(*args, ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha="center", color="red")
        fig.tight_layout()
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        plt.close(fig)
        saved.append(path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from bench_latency.py JSONL output."
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to JSONL file from bench_latency.py")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots"), help="Output directory for plots"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png", help="Output image format"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster formats")
    parser.add_argument("--dark-theme", action="store_true", help="Use dark background theme")
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        print(f"Error: {args.input_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    setup_style(args.dark_theme)

    rows = load_jsonl(args.input_jsonl)
    calls, summaries, sweeps = split_rows(rows)
    print(
        f"Loaded {len(rows)} rows: {len(calls)} calls, {len(summaries)} run_summaries, {len(sweeps)} sweep_calls",
        flush=True,
    )

    saved = make_all_plots(rows, args.output_dir, fmt=args.format, dpi=args.dpi)
    for p in saved:
        print(f"  {p}", flush=True)
    print(f"Done: {len(saved)} plots in {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
