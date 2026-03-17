"""Metric specifications, derived computations, and episode-derived signals."""

from __future__ import annotations

from dataclasses import dataclass

from .loaders import RunData


# ---------------------------------------------------------------------------
# SignalSpec: curated metric key mappings
# ---------------------------------------------------------------------------


@dataclass
class SignalSpec:
    name: str  # display name (e.g. "truth_win|disagr")
    key: str  # metrics.jsonl key (e.g. "env/all/truth_win_if_disagreement")
    higher_is_better: bool | None  # True, False, or None (neutral)
    fmt: str = ".3f"  # format string for value display
    derived: bool = False  # if True, computed from row or episodes, not a raw key


def _s(name: str, key: str, hib: bool | None, fmt: str = ".3f") -> SignalSpec:
    """Shorthand for a raw (non-derived) signal spec."""
    return SignalSpec(name=name, key=key, higher_is_better=hib, fmt=fmt)


def _d(name: str, hib: bool | None, fmt: str = ".3f") -> SignalSpec:
    """Shorthand for a derived signal spec (no raw key)."""
    return SignalSpec(name=name, key="", higher_is_better=hib, fmt=fmt, derived=True)


# -- Gate: SIGNAL --
SIGNAL_PRIMARY: list[SignalSpec] = [
    _s("truth_win|disagr", "env/all/truth_win_if_disagreement", True),
    _s("reward", "env/all/reward/total", True),
    _s("kl", "optim/kl_sample_train_v2", False),
    _s("entropy", "optim/entropy", None),
]
SIGNAL_SECONDARY: list[SignalSpec] = [
    _d("signal_density", True),
    _d("accuracy_mean", True),
    _s("truth_surfaced", "env/all/truth_surfaced", True),
    _s("frac_all_bad", "env/all/by_group/frac_all_bad", False),
]

# -- Gate: JUDGE --
JUDGE_PRIMARY: list[SignalSpec] = [
    _s("judge_quality", "env/all/judge_quality", True),
    _d("judge_exploitation", False),
    _d("bullshit_contest", False),
]
JUDGE_SECONDARY: list[SignalSpec] = [
    _d("wrong_wins_mean", False),
    _d("concession_correctness_mean", True),
]

# -- Gate: SYMMETRY --
SYMMETRY_PRIMARY: list[SignalSpec] = [
    _d("seat_bias", False),
    _d("reward_gap", False),
    _d("sycophancy_rate", False),
    _d("agree_B_win", False),
]
SYMMETRY_SECONDARY: list[SignalSpec] = [
    _d("accuracy_gap", None),
    _d("stance_ratio", None),
    _d("parse_success_gap", None),
]

# -- Gate: QUALITY --
QUALITY_PRIMARY: list[SignalSpec] = [
    _s("parse_success", "env/all/parse_success", True),
    _s("disagreement", "env/all/disagreement", True),
    _s("frac_all_good", "env/all/by_group/frac_all_good", True),
    _s("think_block_rate", "env/all/think_block_rate.debater_a", True),
]
QUALITY_SECONDARY: list[SignalSpec] = [
    _s("frac_mixed", "env/all/by_group/frac_mixed", None),
    _s("total_episodes", "env/all/total_episodes", None, fmt=".0f"),
    _s("convergence_round", "env/all/convergence_round", None),
]

# -- Gate: BUDGET --
BUDGET_PRIMARY: list[SignalSpec] = [
    _s("total_cost_usd", "usage/total_cost_usd", None, fmt="$.2f"),
    _s("time/total", "time/total", None, fmt=".0f"),
    _s("ac_tokens_per_turn", "env/all/ac_tokens_per_turn", None, fmt=".0f"),
    _d("token_growth_rate", False, fmt=".1%"),
]
BUDGET_SECONDARY: list[SignalSpec] = [
    _s("time/train", "time/train", None, fmt=".0f"),
    _s("time/run_evals", "time/run_evals", None, fmt=".0f"),
    _s("ob_tokens_per_turn", "env/all/ob_tokens_per_turn", None, fmt=".0f"),
]

ALL_GATES: dict[str, tuple[list[SignalSpec], list[SignalSpec]]] = {
    "signal": (SIGNAL_PRIMARY, SIGNAL_SECONDARY),
    "judge": (JUDGE_PRIMARY, JUDGE_SECONDARY),
    "symmetry": (SYMMETRY_PRIMARY, SYMMETRY_SECONDARY),
    "quality": (QUALITY_PRIMARY, QUALITY_SECONDARY),
    "budget": (BUDGET_PRIMARY, BUDGET_SECONDARY),
}

ALL_SPECS: list[SignalSpec] = []
for _pri, _sec in ALL_GATES.values():
    ALL_SPECS.extend(_pri)
    ALL_SPECS.extend(_sec)


# ---------------------------------------------------------------------------
# Part 2: Derived metrics from metrics.jsonl rows
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


def seat_bias(row: dict) -> float | None:
    a = _get(row, "env/all/win_rate.debater_a")
    b = _get(row, "env/all/win_rate.debater_b")
    if a is None or b is None:
        return None
    return b - a


def signal_density(row: dict) -> float | None:
    dr = _get(row, "env/all/draw_rate")
    if dr is None:
        return None
    return 1.0 - dr


def accuracy_mean(row: dict) -> float | None:
    a = _get(row, "env/all/accuracy.debater_a")
    b = _get(row, "env/all/accuracy.debater_b")
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def accuracy_gap(row: dict) -> float | None:
    a = _get(row, "env/all/accuracy.debater_a")
    b = _get(row, "env/all/accuracy.debater_b")
    if a is None or b is None:
        return None
    return b - a


def judge_exploitation(row: dict) -> float | None:
    ww_a = _get(row, "env/all/wrong_and_wins.debater_a")
    ww_b = _get(row, "env/all/wrong_and_wins.debater_b")
    cw_a = _get(row, "env/all/correct_and_wins.debater_a")
    cw_b = _get(row, "env/all/correct_and_wins.debater_b")
    if any(v is None for v in (ww_a, ww_b, cw_a, cw_b)):
        return None
    wrong_wins = (ww_a + ww_b) / 2.0
    correct_wins = (cw_a + cw_b) / 2.0
    denom = wrong_wins + correct_wins
    if denom == 0:
        return None
    return wrong_wins / denom


def wrong_wins_mean(row: dict) -> float | None:
    a = _get(row, "env/all/wrong_and_wins.debater_a")
    b = _get(row, "env/all/wrong_and_wins.debater_b")
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def concession_correctness_mean(row: dict) -> float | None:
    a = _get(row, "env/all/concession_correctness.debater_a")
    b = _get(row, "env/all/concession_correctness.debater_b")
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def token_growth_rate(row: dict, first_row: dict) -> float | None:
    now = _get(row, "env/all/ac_tokens_per_turn")
    start = _get(first_row, "env/all/ac_tokens_per_turn")
    if now is None or start is None or start == 0:
        return None
    return now / start


def parse_success_gap(row: dict) -> float | None:
    a = _get(row, "env/all/parse_success.debater_a")
    b = _get(row, "env/all/parse_success.debater_b")
    if a is None or b is None:
        return None
    return b - a


# Row-derived signal dispatch: name -> callable(row, first_row?) -> float|None
_ROW_DERIVED: dict[str, object] = {
    "seat_bias": seat_bias,
    "signal_density": signal_density,
    "accuracy_mean": accuracy_mean,
    "accuracy_gap": accuracy_gap,
    "judge_exploitation": judge_exploitation,
    "wrong_wins_mean": wrong_wins_mean,
    "concession_correctness_mean": concession_correctness_mean,
    "token_growth_rate": token_growth_rate,
    "parse_success_gap": parse_success_gap,
}


# ---------------------------------------------------------------------------
# Part 3: Episode-derived signals
# ---------------------------------------------------------------------------


def sycophancy_rate(episodes: list[dict]) -> tuple[float | None, int, int]:
    """Among concessions (stance_change=1) where the OTHER debater was wrong,
    what fraction conceded? Returns (rate, n_sycophantic, n_concessions)."""
    n_sycophantic = 0
    n_concessions = 0
    for ep in episodes:
        sigs = ep.get("signals", {})
        role = ep.get("role")
        if role not in ("debater_a", "debater_b"):
            continue
        other = "debater_b" if role == "debater_a" else "debater_a"
        sc = sigs.get(f"stance_change.{role}")
        other_acc = sigs.get(f"accuracy.{other}")
        if sc is None or other_acc is None:
            continue
        if sc > 0:
            n_concessions += 1
            if other_acc == 0:
                n_sycophantic += 1
    if n_concessions == 0:
        return (None, 0, 0)
    return (n_sycophantic / n_concessions, n_sycophantic, n_concessions)


def bullshit_contest_rate(episodes: list[dict]) -> tuple[float | None, int, int]:
    """Among decisive episodes (winner != None and not draw), what fraction
    have both debaters wrong? Returns (rate, n_both_wrong_decisive, n_decisive)."""
    n_both_wrong = 0
    n_decisive = 0
    seen_debates: set[str] = set()
    for ep in episodes:
        debate_id = ep.get("debate_id", "")
        if debate_id in seen_debates:
            continue
        seen_debates.add(debate_id)
        sigs = ep.get("signals", {})
        winner = ep.get("winner")
        draw = sigs.get("draw_rate", 0)
        if winner is None or draw == 1:
            continue
        n_decisive += 1
        acc_a = sigs.get("accuracy.debater_a", 0)
        acc_b = sigs.get("accuracy.debater_b", 0)
        if acc_a == 0 and acc_b == 0:
            n_both_wrong += 1
    if n_decisive == 0:
        return (None, 0, 0)
    return (n_both_wrong / n_decisive, n_both_wrong, n_decisive)


def agree_B_win_rate(episodes: list[dict]) -> tuple[float | None, int, int]:
    """Among episodes where both debaters agree (disagreement=0), what fraction
    does B win? Returns (rate, n_B_wins_agree, n_agree)."""
    n_b_wins = 0
    n_agree = 0
    seen_debates: set[str] = set()
    for ep in episodes:
        debate_id = ep.get("debate_id", "")
        if debate_id in seen_debates:
            continue
        seen_debates.add(debate_id)
        sigs = ep.get("signals", {})
        if sigs.get("disagreement", 1) != 0:
            continue
        n_agree += 1
        winner = ep.get("winner")
        if winner == "debater_b":
            n_b_wins += 1
    if n_agree == 0:
        return (None, 0, 0)
    return (n_b_wins / n_agree, n_b_wins, n_agree)


def reward_gap(episodes: list[dict]) -> tuple[float | None, int, int]:
    """Mean reward for debater_b minus mean reward for debater_a.
    Returns (gap, n_b, n_a)."""
    sum_a, n_a = 0.0, 0
    sum_b, n_b = 0.0, 0
    for ep in episodes:
        role = ep.get("role")
        reward = ep.get("reward")
        if reward is None:
            continue
        if role == "debater_a":
            sum_a += reward
            n_a += 1
        elif role == "debater_b":
            sum_b += reward
            n_b += 1
    if n_a == 0 or n_b == 0:
        return (None, n_b, n_a)
    return (sum_b / n_b - sum_a / n_a, n_b, n_a)


def stance_ratio(episodes: list[dict]) -> tuple[float | None, int, int]:
    """sum(stance_change.a) / sum(stance_change.b). Returns (ratio, sum_a, sum_b)."""
    sum_a, sum_b = 0.0, 0.0
    for ep in episodes:
        sigs = ep.get("signals", {})
        sc_a = sigs.get("stance_change.debater_a")
        sc_b = sigs.get("stance_change.debater_b")
        if sc_a is not None:
            sum_a += sc_a
        if sc_b is not None:
            sum_b += sc_b
    int_a, int_b = int(sum_a), int(sum_b)
    if sum_b == 0:
        return (None, int_a, int_b)
    return (sum_a / sum_b, int_a, int_b)


# Episode-derived signal dispatch
_EPISODE_DERIVED: dict[str, object] = {
    "sycophancy_rate": sycophancy_rate,
    "bullshit_contest": bullshit_contest_rate,
    "agree_B_win": agree_B_win_rate,
    "reward_gap": reward_gap,
    "stance_ratio": stance_ratio,
}


# ---------------------------------------------------------------------------
# Part 4: Compute all signals for a RunData
# ---------------------------------------------------------------------------


@dataclass
class SignalResult:
    """Output of compute_signals: time series + episode denominators."""

    series: dict[str, list[float | None]]
    # For episode-derived signals: (numerator, denominator) counts.
    # Only populated for signals computed from episodes.
    episode_counts: dict[str, tuple[int, int]]


def compute_signals(run: RunData) -> SignalResult:
    """For each curated signal, return the time series (one value per step)."""
    series: dict[str, list[float | None]] = {}
    episode_counts: dict[str, tuple[int, int]] = {}
    rows = run.metrics_rows
    if not rows:
        return SignalResult(series=series, episode_counts=episode_counts)
    first_row = rows[0]

    for spec in ALL_SPECS:
        name = spec.name

        # Raw key lookup
        if not spec.derived:
            series[name] = [_get(row, spec.key) for row in rows]
            continue

        # Row-derived
        if name in _ROW_DERIVED:
            fn = _ROW_DERIVED[name]
            if name == "token_growth_rate":
                series[name] = [fn(row, first_row) for row in rows]
            else:
                series[name] = [fn(row) for row in rows]
            continue

        # Episode-derived: no step field => run-aggregate, single value
        if name in _EPISODE_DERIVED and run.episode_rows is not None:
            fn = _EPISODE_DERIVED[name]
            val, n, N = fn(run.episode_rows)
            # Broadcast as constant across all steps
            series[name] = [val] * len(rows)
            episode_counts[name] = (n, N)
            continue

        # Unknown derived signal — fill with None
        series[name] = [None] * len(rows)

    return SignalResult(series=series, episode_counts=episode_counts)
