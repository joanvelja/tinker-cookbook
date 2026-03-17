"""5-gate health evaluation for debate training runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .signals import SignalResult


class GateStatus(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class GateResult:
    name: str  # "SIGNAL", "JUDGE", "SYMMETRY", "QUALITY", "BUDGET"
    status: GateStatus  # worst of all metric statuses in this gate
    streak: int  # consecutive steps at current status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_SEVERITY = {GateStatus.OK: 0, GateStatus.WARN: 1, GateStatus.FAIL: 2}


def _worst(a: GateStatus, b: GateStatus) -> GateStatus:
    return a if _STATUS_SEVERITY[a] >= _STATUS_SEVERITY[b] else b


def _last(series: list[float | None]) -> float | None:
    """Return the last non-None value, or None if empty."""
    for v in reversed(series):
        if v is not None:
            return v
    return None


def _count_trailing_flatline(series: list[float | None], threshold: float = 0.001) -> int:
    """Count consecutive steps from the end where abs(value) < threshold."""
    count = 0
    for v in reversed(series):
        if v is not None and abs(v) < threshold:
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Per-gate evaluation (returns status at each step for streak calculation)
# ---------------------------------------------------------------------------


def _check_threshold(
    value: float | None,
    *,
    fail_below: float | None = None,
    warn_below: float | None = None,
    fail_above: float | None = None,
    warn_above: float | None = None,
    fail_abs_above: float | None = None,
    warn_abs_above: float | None = None,
) -> GateStatus:
    """Evaluate a single value against thresholds. Returns status."""
    if value is None:
        return GateStatus.OK  # missing data = no judgment
    if fail_below is not None and value < fail_below:
        return GateStatus.FAIL
    if fail_above is not None and value > fail_above:
        return GateStatus.FAIL
    if fail_abs_above is not None and abs(value) > fail_abs_above:
        return GateStatus.FAIL
    if warn_below is not None and value < warn_below:
        return GateStatus.WARN
    if warn_above is not None and value > warn_above:
        return GateStatus.WARN
    if warn_abs_above is not None and abs(value) > warn_abs_above:
        return GateStatus.WARN
    return GateStatus.OK


def _gate_signal_at_step(signals: SignalResult, step: int) -> GateStatus:
    """Evaluate SIGNAL gate at a given step index."""
    s = GateStatus.OK

    # truth_win_if_disagreement: FAIL < 0.40, WARN < 0.55
    tw = signals.series.get("truth_win|disagr", [])
    if step < len(tw):
        s = _worst(s, _check_threshold(tw[step], fail_below=0.40, warn_below=0.55))

    # reward/total flatline: FAIL if abs < 0.001 for 3+ consecutive steps ending at `step`
    rw = signals.series.get("reward", [])
    if step < len(rw):
        # Check flatline up to this step
        count = 0
        for i in range(step, -1, -1):
            v = rw[i]
            if v is not None and abs(v) < 0.001:
                count += 1
            else:
                break
        if count >= 3:
            s = _worst(s, GateStatus.FAIL)

    # kl_v2: FAIL > 0.010, WARN > 0.005
    kl = signals.series.get("kl", [])
    if step < len(kl):
        s = _worst(s, _check_threshold(kl[step], fail_above=0.010, warn_above=0.005))

    # entropy: FAIL > 0.80, WARN > 0.65
    ent = signals.series.get("entropy", [])
    if step < len(ent):
        s = _worst(s, _check_threshold(ent[step], fail_above=0.80, warn_above=0.65))

    return s


def _gate_judge_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # judge_quality: FAIL < 0.25, WARN < 0.35
    jq = signals.series.get("judge_quality", [])
    if step < len(jq):
        s = _worst(s, _check_threshold(jq[step], fail_below=0.25, warn_below=0.35))

    # judge_exploitation: FAIL > 0.60, WARN > 0.45
    je = signals.series.get("judge_exploitation", [])
    if step < len(je):
        s = _worst(s, _check_threshold(je[step], fail_above=0.60, warn_above=0.45))

    # bullshit_contest: FAIL > 0.60, WARN > 0.45
    bc = signals.series.get("bullshit_contest", [])
    if step < len(bc):
        s = _worst(s, _check_threshold(bc[step], fail_above=0.60, warn_above=0.45))

    return s


def _gate_symmetry_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # seat_bias (abs): FAIL > 0.40, WARN > 0.25
    sb = signals.series.get("seat_bias", [])
    if step < len(sb):
        s = _worst(s, _check_threshold(sb[step], fail_abs_above=0.40, warn_abs_above=0.25))

    # reward_gap (abs): FAIL > 0.80, WARN > 0.50
    rg = signals.series.get("reward_gap", [])
    if step < len(rg):
        s = _worst(s, _check_threshold(rg[step], fail_abs_above=0.80, warn_abs_above=0.50))

    # sycophancy_rate: FAIL > 0.80, WARN > 0.65
    sr = signals.series.get("sycophancy_rate", [])
    if step < len(sr):
        s = _worst(s, _check_threshold(sr[step], fail_above=0.80, warn_above=0.65))

    return s


def _gate_quality_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # parse_success: FAIL < 0.85, WARN < 0.95
    ps = signals.series.get("parse_success", [])
    if step < len(ps):
        s = _worst(s, _check_threshold(ps[step], fail_below=0.85, warn_below=0.95))

    # disagreement: FAIL < 0.05, WARN < 0.10
    dg = signals.series.get("disagreement", [])
    if step < len(dg):
        s = _worst(s, _check_threshold(dg[step], fail_below=0.05, warn_below=0.10))

    return s


def _gate_budget_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # ac_tokens_per_turn: FAIL > 4000, WARN > 3000
    at = signals.series.get("ac_tokens_per_turn", [])
    if step < len(at):
        s = _worst(s, _check_threshold(at[step], fail_above=4000, warn_above=3000))

    # token_growth_rate: FAIL > 2.0, WARN > 1.5
    tg = signals.series.get("token_growth_rate", [])
    if step < len(tg):
        s = _worst(s, _check_threshold(tg[step], fail_above=2.0, warn_above=1.5))

    return s


# ---------------------------------------------------------------------------
# Streak calculation
# ---------------------------------------------------------------------------

_GATE_STEP_FNS = {
    "SIGNAL": _gate_signal_at_step,
    "JUDGE": _gate_judge_at_step,
    "SYMMETRY": _gate_symmetry_at_step,
    "QUALITY": _gate_quality_at_step,
    "BUDGET": _gate_budget_at_step,
}


def _evaluate_gate(name: str, signals: SignalResult) -> GateResult:
    """Evaluate a gate across all steps, return final status + streak."""
    n_steps = 0
    for vals in signals.series.values():
        n_steps = max(n_steps, len(vals))

    if n_steps == 0:
        return GateResult(name=name, status=GateStatus.OK, streak=0)

    step_fn = _GATE_STEP_FNS[name]
    statuses = [step_fn(signals, i) for i in range(n_steps)]

    final_status = statuses[-1]
    streak = 1
    for i in range(len(statuses) - 2, -1, -1):
        if statuses[i] == final_status:
            streak += 1
        else:
            break

    return GateResult(name=name, status=final_status, streak=streak)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_gates(signals: SignalResult) -> list[GateResult]:
    """Evaluate all 5 gates. Returns list ordered: SIGNAL, JUDGE, SYMMETRY, QUALITY, BUDGET."""
    return [
        _evaluate_gate("SIGNAL", signals),
        _evaluate_gate("JUDGE", signals),
        _evaluate_gate("SYMMETRY", signals),
        _evaluate_gate("QUALITY", signals),
        _evaluate_gate("BUDGET", signals),
    ]
