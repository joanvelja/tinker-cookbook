"""Signal specs and derived computations for RLVR training monitoring."""

from __future__ import annotations

from dataclasses import dataclass

from .loaders import RunData


# ---------------------------------------------------------------------------
# SignalSpec
# ---------------------------------------------------------------------------


@dataclass
class SignalSpec:
    name: str  # display name
    key: str  # metrics.jsonl key (empty if derived)
    higher_is_better: bool | None
    fmt: str = ".3f"
    derived: bool = False


def _s(name: str, key: str, hib: bool | None, fmt: str = ".3f") -> SignalSpec:
    return SignalSpec(name=name, key=key, higher_is_better=hib, fmt=fmt)


def _d(name: str, hib: bool | None, fmt: str = ".3f") -> SignalSpec:
    return SignalSpec(name=name, key="", higher_is_better=hib, fmt=fmt, derived=True)


# -- Gate: LEARNING --
LEARNING_PRIMARY: list[SignalSpec] = [
    _s("correct", "env/all/correct", True),
    _s("reward", "env/all/reward/total", True),
    _d("correct_slope", True),
]
LEARNING_SECONDARY: list[SignalSpec] = [
    _s("frac_all_good", "env/all/by_group/frac_all_good", True),
    _s("frac_all_bad", "env/all/by_group/frac_all_bad", False),
    _s("frac_mixed", "env/all/by_group/frac_mixed", None),
]

# -- Gate: FORMAT --
FORMAT_PRIMARY: list[SignalSpec] = [
    _s("format_boxed", "env/all/format_boxed", True),
    _s("format_eos", "env/all/format_eos", True),
    _d("truncation_rate", False),
    _d("extract_fail_rate", False),
]
FORMAT_SECONDARY: list[SignalSpec] = [
    # Older RLVR runs use "env/all/format" instead of format_boxed
    _s("format_legacy", "env/all/format", True),
]

# -- Gate: COMPRESSION --
COMPRESSION_PRIMARY: list[SignalSpec] = [
    _s("tokens/resp", "env/all/ac_tokens_per_turn", None, fmt=".0f"),
    _d("tokens_slope", False),
]
COMPRESSION_SECONDARY: list[SignalSpec] = [
    _s("entropy", "optim/entropy", None),
    _s("ob_tokens", "env/all/ob_tokens_per_turn", None, fmt=".0f"),
]

# -- Gate: GRADER --
# grade_status (ok/error/timeout) lives in logtree logs, not numeric metrics.
# The only grader health signal in metrics.jsonl is check_answer_s latency.
GRADER_PRIMARY: list[SignalSpec] = [
    _s("check_answer_s", "env/all/time/check_answer_s", False),
]
GRADER_SECONDARY: list[SignalSpec] = [
    _s("kl", "optim/kl_sample_train_v2", False),
]

# -- Gate: OVERFIT --
OVERFIT_PRIMARY: list[SignalSpec] = [
    _d("train_eval_gap", False),
    _s("eval_correct", "test/test/env/all/correct", True),
]
OVERFIT_SECONDARY: list[SignalSpec] = [
    _s("eval_format", "test/test/env/all/format_boxed", True),
    _s("eval_reward", "test/test/env/all/reward/total", True),
]

ALL_GATES: dict[str, tuple[list[SignalSpec], list[SignalSpec]]] = {
    "learning": (LEARNING_PRIMARY, LEARNING_SECONDARY),
    "format": (FORMAT_PRIMARY, FORMAT_SECONDARY),
    "compression": (COMPRESSION_PRIMARY, COMPRESSION_SECONDARY),
    "grader": (GRADER_PRIMARY, GRADER_SECONDARY),
    "overfit": (OVERFIT_PRIMARY, OVERFIT_SECONDARY),
}

ALL_SPECS: list[SignalSpec] = []
for _pri, _sec in ALL_GATES.values():
    ALL_SPECS.extend(_pri)
    ALL_SPECS.extend(_sec)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(row: dict, key: str, default: float | None = None) -> float | None:
    v = row.get(key)
    if v is None:
        return default
    try:
        f = float(v)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def _slope(values: list[float | None], window: int = 10) -> float | None:
    """Linear slope over last `window` non-None values."""
    clean = [v for v in values if v is not None]
    tail = clean[-window:] if len(clean) >= window else clean
    if len(tail) < 2:
        return None
    n = len(tail)
    x_mean = (n - 1) / 2.0
    y_mean = sum(tail) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(tail))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Derived signal computations
# ---------------------------------------------------------------------------


def correct_slope(rows: list[dict]) -> list[float | None]:
    """Rolling slope of correct rate over last 10 steps."""
    series = [_get(row, "env/all/correct") for row in rows]
    result: list[float | None] = []
    for i in range(len(rows)):
        result.append(_slope(series[: i + 1], window=10))
    return result


def tokens_slope(rows: list[dict]) -> list[float | None]:
    """Rolling slope of tokens/response over last 10 steps."""
    series = [_get(row, "env/all/ac_tokens_per_turn") for row in rows]
    result: list[float | None] = []
    for i in range(len(rows)):
        result.append(_slope(series[: i + 1], window=10))
    return result


def truncation_rate(rows: list[dict]) -> list[float | None]:
    """1 - format_eos gives truncation rate."""
    result: list[float | None] = []
    for row in rows:
        eos = _get(row, "env/all/format_eos")
        if eos is None:
            # Fall back to legacy format key
            fmt = _get(row, "env/all/format")
            result.append(1.0 - fmt if fmt is not None else None)
        else:
            result.append(1.0 - eos)
    return result


def extract_fail_rate(rows: list[dict]) -> list[float | None]:
    """Fraction of responses where answer extraction failed (1 - format_boxed).

    This measures format extraction failures (model didn't produce \boxed{} tags),
    NOT grader errors. Grader health (grade_status) is in logtree logs, not metrics.
    """
    result: list[float | None] = []
    for row in rows:
        boxed = _get(row, "env/all/format_boxed")
        if boxed is None:
            fmt = _get(row, "env/all/format")
            result.append(1.0 - fmt if fmt is not None else None)
        else:
            result.append(1.0 - boxed)
    return result


def train_eval_gap(rows: list[dict]) -> list[float | None]:
    """Train correct - eval correct. Positive = potential overfitting.

    Only computed at steps where eval data actually exists to avoid
    false positives from forward-filling stale eval values.
    """
    result: list[float | None] = []
    for row in rows:
        eval_correct = _get(row, "test/test/env/all/correct")
        train_correct = _get(row, "env/all/correct")
        if eval_correct is not None and train_correct is not None:
            result.append(train_correct - eval_correct)
        else:
            result.append(None)
    return result


# Dispatch table for derived signals
_DERIVED_FNS: dict[str, object] = {
    "correct_slope": correct_slope,
    "tokens_slope": tokens_slope,
    "truncation_rate": truncation_rate,
    "extract_fail_rate": extract_fail_rate,
    "train_eval_gap": train_eval_gap,
}


# ---------------------------------------------------------------------------
# Compute all signals
# ---------------------------------------------------------------------------


@dataclass
class SignalResult:
    series: dict[str, list[float | None]]


def _dedup_rows_by_step(rows: list[dict]) -> list[dict]:
    """Keep last entry per step, preserving order. Handles resume re-evals."""
    by_step: dict[int, dict] = {}
    for row in rows:
        step = row.get("step")
        if step is None:
            continue
        if step in by_step:
            # Merge: keep all keys, later values overwrite
            by_step[step].update(row)
        else:
            by_step[step] = dict(row)
    return [by_step[s] for s in sorted(by_step)]


def compute_signals(run: RunData) -> SignalResult:
    series: dict[str, list[float | None]] = {}
    rows = _dedup_rows_by_step(run.metrics_rows)
    if not rows:
        return SignalResult(series=series)

    for spec in ALL_SPECS:
        name = spec.name

        if not spec.derived:
            series[name] = [_get(row, spec.key) for row in rows]
            continue

        if name in _DERIVED_FNS:
            fn = _DERIVED_FNS[name]
            series[name] = fn(rows)
            continue

        series[name] = [None] * len(rows)

    return SignalResult(series=series)
