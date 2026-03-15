"""Terminal rendering for spot-check: sparklines, gate blocks, full run output."""

from __future__ import annotations

from rich.text import Text

from .gates import GateResult, GateStatus
from .signals import ALL_GATES, SignalResult, SignalSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARK_CHARS = "▁▂▃▄▅▆▇█"

_STATUS_STYLE: dict[GateStatus, tuple[str, str]] = {
    GateStatus.OK: ("green", " OK "),
    GateStatus.WARN: ("yellow", "WARN"),
    GateStatus.FAIL: ("red", "FAIL"),
}


# ---------------------------------------------------------------------------
# Sparkline primitives (ported from viz_prototype_v2.py)
# ---------------------------------------------------------------------------


def sparkline(values: list[float | None], width: int = 10) -> str:
    """Render a sparkline string with per-metric local min-max scaling."""
    clean = [v for v in values if v is not None]
    if not clean:
        return " " * width
    if len(clean) <= width:
        sampled = clean
    else:
        indices = [int(i * (len(clean) - 1) / (width - 1)) for i in range(width)]
        sampled = [clean[i] for i in indices]
    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi - lo > 1e-9 else 1e-9
    out = []
    for v in sampled:
        idx = int((v - lo) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def is_stable(values: list[float | None], threshold: float = 0.02) -> bool:
    """True if max - min < threshold across all non-None values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return True
    return (max(clean) - min(clean)) < threshold


def recent_trend(values: list[float | None], window: int = 5) -> float:
    """Average slope over last `window` non-None values."""
    clean = [v for v in values if v is not None]
    tail = clean[-window:] if len(clean) >= window else clean
    if len(tail) < 2:
        return 0.0
    slopes = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    return sum(slopes) / len(slopes)


def streak(values: list[float | None]) -> tuple[str, int]:
    """Direction + count of consecutive same-direction moves at end of series."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return ("→", 0)
    direction = None
    count = 0
    for i in range(len(clean) - 1, 0, -1):
        d = clean[i] - clean[i - 1]
        if abs(d) < 1e-6:
            cur = "→"
        elif d > 0:
            cur = "↑"
        else:
            cur = "↓"
        if direction is None:
            direction = cur
            count = 1
        elif cur == direction:
            count += 1
        else:
            break
    return (direction or "→", count)


def trend_color(trend: float, higher_is_better: bool | None) -> str:
    """Return Rich style name based on trend direction and polarity."""
    if abs(trend) < 0.003:
        return "yellow"
    if higher_is_better is None:
        return "yellow"
    improving = (trend > 0) == higher_is_better
    return "green" if improving else "red"


# ---------------------------------------------------------------------------
# Per-metric line rendering (E2 format)
# ---------------------------------------------------------------------------


def _format_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "   —   "
    if fmt.startswith("$"):
        return f"${value:{fmt[1:]}}"
    return f"{value:{fmt}}"


def render_metric_line(
    spec: SignalSpec,
    series: list[float | None],
    episode_counts: dict[str, tuple[int, int]],
) -> Text:
    """Render one metric line in E2 format."""
    clean = [v for v in series if v is not None]
    line = Text()

    # Name
    line.append(f"    {spec.name:<20} ")

    # Current value
    cur = clean[-1] if clean else None
    line.append(f"{_format_value(cur, spec.fmt):>7} ")

    # Optional (n/N) for episode-derived signals
    if spec.name in episode_counts:
        n, N = episode_counts[spec.name]
        line.append(f"({n}/{N})", style="dim")
        line.append(" ")
    else:
        line.append("        ")

    if not clean or len(clean) < 2:
        line.append("          ", style="dim")
        return line

    # Stable metrics: compressed rendering
    if is_stable(series):
        mean = sum(clean) / len(clean)
        std = (sum((v - mean) ** 2 for v in clean) / len(clean)) ** 0.5
        line.append("━━━━━━━━━━", style="dim")
        line.append(f" ±{std:.3f}", style="dim")
        return line

    # Sparkline with trend color
    rt = recent_trend(series)
    color = trend_color(rt, spec.higher_is_better)
    spark = sparkline(series, 10)
    line.append(spark, style=color)
    line.append(" ")

    # Delta since start
    delta = clean[-1] - clean[0]
    if abs(delta) < 0.001:
        line.append("  ≈0  ", style="dim")
    else:
        sign = "+" if delta > 0 else ""
        d_improving = (
            (delta > 0) == spec.higher_is_better if spec.higher_is_better is not None else None
        )
        d_style = "green" if d_improving is True else "red" if d_improving is False else "yellow"
        line.append(f"{sign}{delta:.3f}", style=d_style)

    line.append(" ")

    # Streak
    direction, count = streak(series)
    if count >= 2:
        s_style = "green" if direction == "↑" else "red" if direction == "↓" else "dim"
        if spec.higher_is_better is not None and not spec.higher_is_better:
            # Flip color for lower-is-better metrics
            if s_style == "green":
                s_style = "red"
            elif s_style == "red":
                s_style = "green"
        line.append(f"{direction}{count}", style=s_style)

    return line


# ---------------------------------------------------------------------------
# Gate block rendering
# ---------------------------------------------------------------------------


def render_gate(
    gate: GateResult,
    specs: list[SignalSpec],
    signals: SignalResult,
) -> Text:
    """Render one gate block: header line + metric lines."""
    block = Text()

    # Header: "  SIGNAL     FAIL  ↓4"
    style, label = _STATUS_STYLE.get(gate.status, ("white", " ?? "))
    block.append(f"  {gate.name:<10} ", style="bold")
    block.append(f" {label} ", style=f"bold reverse {style}")
    if gate.streak >= 2:
        if gate.status == GateStatus.OK:
            s_style = "green"
        elif gate.status == GateStatus.FAIL:
            s_style = "red"
        else:
            s_style = "yellow"
        block.append(f"  {gate.streak}×", style=s_style)
    block.append("\n")

    # Metric lines
    for spec in specs:
        series = signals.series.get(spec.name, [])
        line = render_metric_line(spec, series, signals.episode_counts)
        block.append_text(line)
        block.append("\n")

    return block


# ---------------------------------------------------------------------------
# Full run rendering
# ---------------------------------------------------------------------------


def render_run(
    run_name: str,
    gates: list[GateResult],
    signals: SignalResult,
    verbose: bool = False,
) -> Text:
    """Render full spot-check output for one run."""
    out = Text()

    # Step count from signal series lengths
    step_count = 0
    for series in signals.series.values():
        if series:
            step_count = max(step_count, len(series))

    # Header bar
    out.append("═" * 70 + "\n", style="bold")
    out.append(f"  {run_name}", style="bold")
    out.append(f"  step {step_count}", style="dim")
    out.append("\n")
    out.append("─" * 70 + "\n", style="dim")

    # Build gate lookup (gates.py uses uppercase names, signals.py uses lowercase keys)
    gate_lookup = {g.name.lower(): g for g in gates}

    # Render each gate
    for gate_key, (primary, secondary) in ALL_GATES.items():
        gate = gate_lookup.get(gate_key)
        if gate is None:
            continue
        specs = list(primary)
        if verbose:
            specs.extend(secondary)
        block = render_gate(gate, specs, signals)
        out.append_text(block)
        out.append("\n")

    return out
