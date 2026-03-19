"""Terminal rendering for RLVR spot-check: sparklines, gate blocks, full run output."""

from __future__ import annotations

from rich.text import Text

from .gates import GateResult, GateStatus
from .signals import ALL_GATES, SignalResult, SignalSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

_STATUS_STYLE: dict[GateStatus, tuple[str, str]] = {
    GateStatus.OK: ("green", " OK "),
    GateStatus.WARN: ("yellow", "WARN"),
    GateStatus.FAIL: ("red", "FAIL"),
}


# ---------------------------------------------------------------------------
# Sparkline primitives (shared pattern with spot_check)
# ---------------------------------------------------------------------------


def sparkline(values: list[float | None], width: int = 10) -> str:
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
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return True
    return (max(clean) - min(clean)) < threshold


def recent_trend(values: list[float | None], window: int = 5) -> float:
    clean = [v for v in values if v is not None]
    tail = clean[-window:] if len(clean) >= window else clean
    if len(tail) < 2:
        return 0.0
    slopes = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    return sum(slopes) / len(slopes)


def streak(values: list[float | None]) -> tuple[str, int]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return ("\u2192", 0)
    direction = None
    count = 0
    for i in range(len(clean) - 1, 0, -1):
        d = clean[i] - clean[i - 1]
        if abs(d) < 1e-6:
            cur = "\u2192"
        elif d > 0:
            cur = "\u2191"
        else:
            cur = "\u2193"
        if direction is None:
            direction = cur
            count = 1
        elif cur == direction:
            count += 1
        else:
            break
    return (direction or "\u2192", count)


def trend_color(trend: float, higher_is_better: bool | None) -> str:
    if abs(trend) < 0.003:
        return "yellow"
    if higher_is_better is None:
        return "yellow"
    improving = (trend > 0) == higher_is_better
    return "green" if improving else "red"


# ---------------------------------------------------------------------------
# Per-metric line rendering
# ---------------------------------------------------------------------------


def _format_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "   \u2014   "
    if fmt.startswith("$"):
        return f"${value:{fmt[1:]}}"
    return f"{value:{fmt}}"


def render_metric_line(spec: SignalSpec, series: list[float | None]) -> Text:
    clean = [v for v in series if v is not None]
    line = Text()

    line.append(f"    {spec.name:<20} ")

    cur = clean[-1] if clean else None
    line.append(f"{_format_value(cur, spec.fmt):>7} ")
    line.append("        ")

    if not clean or len(clean) < 2:
        line.append("          ", style="dim")
        return line

    if is_stable(series):
        mean = sum(clean) / len(clean)
        std = (sum((v - mean) ** 2 for v in clean) / len(clean)) ** 0.5
        line.append("\u2501" * 10, style="dim")
        line.append(f" \u00b1{std:.3f}", style="dim")
        return line

    rt = recent_trend(series)
    color = trend_color(rt, spec.higher_is_better)
    spark = sparkline(series, 10)
    line.append(spark, style=color)
    line.append(" ")

    delta = clean[-1] - clean[0]
    if abs(delta) < 0.001:
        line.append("  \u22480  ", style="dim")
    else:
        sign = "+" if delta > 0 else ""
        d_improving = (
            (delta > 0) == spec.higher_is_better if spec.higher_is_better is not None else None
        )
        d_style = "green" if d_improving is True else "red" if d_improving is False else "yellow"
        line.append(f"{sign}{delta:.3f}", style=d_style)

    line.append(" ")

    direction, count = streak(series)
    if count >= 2:
        s_style = "green" if direction == "\u2191" else "red" if direction == "\u2193" else "dim"
        if spec.higher_is_better is not None and not spec.higher_is_better:
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
    block = Text()

    style, label = _STATUS_STYLE.get(gate.status, ("white", " ?? "))
    block.append(f"  {gate.name:<12} ", style="bold")
    block.append(f" {label} ", style=f"bold reverse {style}")
    if gate.streak >= 2:
        if gate.status == GateStatus.OK:
            s_style = "green"
        elif gate.status == GateStatus.FAIL:
            s_style = "red"
        else:
            s_style = "yellow"
        block.append(f"  {gate.streak}\u00d7", style=s_style)
    block.append("\n")

    for spec in specs:
        series = signals.series.get(spec.name, [])
        line = render_metric_line(spec, series)
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
    out = Text()

    last_step = max((len(s) - 1 for s in signals.series.values() if s), default=0)

    out.append("\u2550" * 70 + "\n", style="bold")
    out.append(f"  {run_name}", style="bold")
    out.append(f"  step {last_step}", style="dim")
    out.append("\n")
    out.append("\u2500" * 70 + "\n", style="dim")

    gate_lookup = {g.name.lower(): g for g in gates}

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
