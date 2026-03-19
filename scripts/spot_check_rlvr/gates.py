"""5-gate health evaluation for RLVR training runs."""

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
    name: str
    status: GateStatus
    streak: int


_STATUS_SEVERITY = {GateStatus.OK: 0, GateStatus.WARN: 1, GateStatus.FAIL: 2}


def _worst(a: GateStatus, b: GateStatus) -> GateStatus:
    return a if _STATUS_SEVERITY[a] >= _STATUS_SEVERITY[b] else b


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
    if value is None:
        return GateStatus.OK
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


# ---------------------------------------------------------------------------
# Per-gate evaluation at a given step
# ---------------------------------------------------------------------------


def _gate_learning_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # correct_slope: GREEN if > 0.001, YELLOW if flat, RED if declining
    # Exception: if correct rate is already >= 0.99, flat slope is OK (can't improve)
    cs = signals.series.get("correct_slope", [])
    cr = signals.series.get("correct", [])
    if step < len(cs) and cs[step] is not None:
        slope = cs[step]
        current_correct = cr[step] if step < len(cr) else None
        already_maxed = current_correct is not None and current_correct >= 0.99
        if slope < -0.001:
            s = _worst(s, GateStatus.FAIL)
        elif slope < 0.001 and not already_maxed:
            s = _worst(s, GateStatus.WARN)

    # reward flatline: FAIL if reward < 0.001 for 3+ consecutive steps
    rw = signals.series.get("reward", [])
    if step < len(rw):
        count = 0
        for i in range(step, -1, -1):
            v = rw[i]
            if v is not None and abs(v) < 0.001:
                count += 1
            else:
                break
        if count >= 3:
            s = _worst(s, GateStatus.FAIL)

    return s


def _gate_format_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # format_boxed: GREEN if > 0.8, YELLOW if > 0.5, RED if <= 0.5
    fb = signals.series.get("format_boxed", [])
    if step < len(fb):
        s = _worst(s, _check_threshold(fb[step], fail_below=0.5, warn_below=0.8))

    # format_eos: GREEN if > 0.8, YELLOW if > 0.5, RED if <= 0.5
    fe = signals.series.get("format_eos", [])
    if step < len(fe):
        s = _worst(s, _check_threshold(fe[step], fail_below=0.5, warn_below=0.8))

    # Fall back to legacy format key if neither boxed/eos have real values
    fb_has_data = any(v is not None for v in fb)
    fe_has_data = any(v is not None for v in fe)
    if not fb_has_data and not fe_has_data:
        fl = signals.series.get("format_legacy", [])
        if step < len(fl):
            s = _worst(s, _check_threshold(fl[step], fail_below=0.5, warn_below=0.8))

    return s


def _gate_compression_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # tokens/resp: FAIL if > 4000, WARN if > 3000
    tr = signals.series.get("tokens/resp", [])
    if step < len(tr):
        s = _worst(s, _check_threshold(tr[step], fail_above=4000, warn_above=3000))

    # tokens_slope: RED if strongly positive (growing responses)
    ts = signals.series.get("tokens_slope", [])
    if step < len(ts) and ts[step] is not None:
        if ts[step] > 50:  # growing by >50 tokens/step
            s = _worst(s, GateStatus.FAIL)
        elif ts[step] > 20:
            s = _worst(s, GateStatus.WARN)

    return s


def _gate_grader_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # check_answer_s latency: FAIL if > 5s, WARN if > 2s
    # This is the only reliable grader health signal in metrics.jsonl.
    # grade_status (ok/error/timeout) lives in logtree logs, not metrics.
    ca = signals.series.get("check_answer_s", [])
    if step < len(ca):
        s = _worst(s, _check_threshold(ca[step], fail_above=5.0, warn_above=2.0))

    return s


def _gate_overfit_at_step(signals: SignalResult, step: int) -> GateStatus:
    s = GateStatus.OK

    # train_eval_gap: GREEN if < 0.05, YELLOW if < 0.10, RED if >= 0.10
    teg = signals.series.get("train_eval_gap", [])
    if step < len(teg):
        s = _worst(s, _check_threshold(teg[step], fail_above=0.10, warn_above=0.05))

    # eval_correct declining: check if last eval is lower than first eval
    ec = signals.series.get("eval_correct", [])
    clean_ec = [(i, v) for i, v in enumerate(ec) if v is not None]
    if len(clean_ec) >= 2 and step >= clean_ec[-1][0]:
        first_val = clean_ec[0][1]
        last_val = clean_ec[-1][1]
        # Eval declining while train improving = bad sign
        train_cs = signals.series.get("correct_slope", [])
        train_improving = (
            step < len(train_cs) and train_cs[step] is not None and train_cs[step] > 0.001
        )
        if last_val < first_val - 0.05 and train_improving:
            s = _worst(s, GateStatus.FAIL)
        elif last_val < first_val - 0.02 and train_improving:
            s = _worst(s, GateStatus.WARN)

    return s


# ---------------------------------------------------------------------------
# Streak + gate evaluation
# ---------------------------------------------------------------------------

_GATE_STEP_FNS = {
    "LEARNING": _gate_learning_at_step,
    "FORMAT": _gate_format_at_step,
    "COMPRESSION": _gate_compression_at_step,
    "GRADER": _gate_grader_at_step,
    "OVERFIT": _gate_overfit_at_step,
}


def _evaluate_gate(name: str, signals: SignalResult) -> GateResult:
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


def evaluate_gates(signals: SignalResult) -> list[GateResult]:
    """Evaluate all 5 RLVR gates."""
    return [
        _evaluate_gate("LEARNING", signals),
        _evaluate_gate("FORMAT", signals),
        _evaluate_gate("COMPRESSION", signals),
        _evaluate_gate("GRADER", signals),
        _evaluate_gate("OVERFIT", signals),
    ]
