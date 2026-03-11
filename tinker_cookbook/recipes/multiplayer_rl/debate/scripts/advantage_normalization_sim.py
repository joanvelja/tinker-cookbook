"""Simulate reward-only normalization schemes for self-play debate rollouts.

The script compares how different group-wise advantage schemes weight
question/side pairs with Bernoulli win rate ``p``. The reported effective
weight ``w(p)`` is the expected scalar multiplying ``∇p`` under a simple
score-function proxy:

    score = 1 / p      if the rollout wins
    score = -1 / (1-p) if the rollout loses

This proxy satisfies:

    E[score] = 0
    E[reward * score] = 1

so ``E[advantage * score]`` is an interpretable per-question weighting.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_P_VALUES = (0.1, 0.2, 0.5, 0.8, 0.9)
DEFAULT_SCHEMES = ("reinforce", "mean_center", "grpo", "maxrl", "power_mean")


@dataclass(frozen=True)
class SimulationRow:
    p: float
    active_group_rate: float
    exact_weight: float
    monte_carlo_weight: float


def _validate_probability(p: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"Expected 0 < p < 1, got {p}")


def _validate_win_count(group_size: int, win_count: int) -> None:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not 0 <= win_count <= group_size:
        raise ValueError(f"win_count must be in [0, {group_size}], got {win_count}")


def winner_loser_advantages(
    group_size: int,
    win_count: int,
    scheme: str,
    *,
    alpha: float = 0.5,
) -> tuple[float, float]:
    """Return the advantage assigned to each winner and loser for a fixed K."""
    _validate_win_count(group_size, win_count)

    mean_reward = win_count / group_size

    if scheme == "reinforce":
        return 1.0, 0.0

    if scheme == "mean_center":
        return 1.0 - mean_reward, -mean_reward

    if scheme == "grpo":
        if win_count in {0, group_size}:
            return 0.0, 0.0
        std_reward = math.sqrt(mean_reward * (1.0 - mean_reward))
        return (1.0 - mean_reward) / std_reward, -mean_reward / std_reward

    if scheme == "maxrl":
        if win_count == 0:
            return 0.0, 0.0
        return (1.0 - mean_reward) / mean_reward, -1.0

    if scheme == "power_mean":
        if win_count == 0:
            return 0.0, 0.0
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        denom = mean_reward**alpha
        return (1.0 - mean_reward) / denom, -mean_reward / denom

    raise ValueError(f"Unknown scheme: {scheme}")


def rollout_advantages_for_win_count(
    group_size: int,
    win_count: int,
    scheme: str,
    *,
    alpha: float = 0.5,
) -> list[float]:
    """Return per-rollout advantages for a group with ``win_count`` wins."""
    winner_advantage, loser_advantage = winner_loser_advantages(
        group_size,
        win_count,
        scheme,
        alpha=alpha,
    )
    return [winner_advantage] * win_count + [loser_advantage] * (group_size - win_count)


def _binomial_probability(group_size: int, win_count: int, p: float) -> float:
    return (
        math.comb(group_size, win_count) * (p**win_count) * ((1.0 - p) ** (group_size - win_count))
    )


def exact_weight(
    p: float,
    group_size: int,
    *,
    scheme: str,
    alpha: float = 0.5,
) -> float:
    """Compute the exact effective weight under a Bernoulli win model."""
    _validate_probability(p)

    total = 0.0
    for win_count in range(group_size + 1):
        winner_advantage, loser_advantage = winner_loser_advantages(
            group_size,
            win_count,
            scheme,
            alpha=alpha,
        )
        score_weight = (
            win_count * winner_advantage / p
            - (group_size - win_count) * loser_advantage / (1.0 - p)
        ) / group_size
        total += _binomial_probability(group_size, win_count, p) * score_weight

    return total


def _vectorized_advantages(
    rewards: np.ndarray,
    *,
    scheme: str,
    alpha: float = 0.5,
) -> np.ndarray:
    counts = rewards.sum(axis=1).astype(int)
    winner_advantages = np.zeros(rewards.shape[0], dtype=np.float64)
    loser_advantages = np.zeros(rewards.shape[0], dtype=np.float64)

    group_size = rewards.shape[1]
    for win_count in range(group_size + 1):
        mask = counts == win_count
        if not np.any(mask):
            continue
        winner_advantage, loser_advantage = winner_loser_advantages(
            group_size,
            win_count,
            scheme,
            alpha=alpha,
        )
        winner_advantages[mask] = winner_advantage
        loser_advantages[mask] = loser_advantage

    return np.where(rewards > 0, winner_advantages[:, None], loser_advantages[:, None])


def monte_carlo_weight(
    p: float,
    group_size: int,
    *,
    scheme: str,
    alpha: float = 0.5,
    trials: int = 200_000,
    seed: int = 0,
) -> float:
    """Estimate the effective weight by simulation."""
    _validate_probability(p)
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")

    rng = np.random.default_rng(seed)
    rewards = rng.binomial(1, p, size=(trials, group_size)).astype(np.float64)
    proxy_scores = np.where(rewards > 0, 1.0 / p, -1.0 / (1.0 - p))
    advantages = _vectorized_advantages(rewards, scheme=scheme, alpha=alpha)
    return float(np.mean(np.mean(advantages * proxy_scores, axis=1)))


def active_group_rate(p: float, group_size: int) -> float:
    """Probability that a group has at least one win and at least one loss."""
    _validate_probability(p)
    return 1.0 - p**group_size - (1.0 - p) ** group_size


def simulate_scheme(
    p_values: Sequence[float],
    group_size: int,
    *,
    scheme: str,
    alpha: float = 0.5,
    trials: int = 200_000,
    seed: int = 0,
) -> list[SimulationRow]:
    return [
        SimulationRow(
            p=p,
            active_group_rate=active_group_rate(p, group_size),
            exact_weight=exact_weight(p, group_size, scheme=scheme, alpha=alpha),
            monte_carlo_weight=monte_carlo_weight(
                p,
                group_size,
                scheme=scheme,
                alpha=alpha,
                trials=trials,
                seed=seed,
            ),
        )
        for p in p_values
    ]


def _parse_probabilities(raw: str) -> tuple[float, ...]:
    p_values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    for p in p_values:
        _validate_probability(p)
    return p_values


def _scheme_label(scheme: str, alpha: float) -> str:
    if scheme == "power_mean":
        return f"power_mean(alpha={alpha:g})"
    return scheme


def _format_float(value: float) -> str:
    return f"{value:>8.3f}"


def _print_weight_table(
    scheme: str,
    rows: Iterable[SimulationRow],
    *,
    alpha: float,
) -> None:
    label = _scheme_label(scheme, alpha)
    print(label)
    print("  p      active     exact    monte_carlo")
    for row in rows:
        print(
            f"  {row.p:0.2f}  "
            f"{_format_float(row.active_group_rate)}  "
            f"{_format_float(row.exact_weight)}  "
            f"{_format_float(row.monte_carlo_weight)}"
        )
    print()


def _print_fixed_group_example(group_size: int, win_count: int, alpha: float) -> None:
    print(f"Fixed-group example: group_size={group_size}, win_count={win_count}")
    print("All reward-only schemes assign the same credit to every winner in the group.")
    print("  scheme                 winner_adv  loser_adv")

    for scheme in DEFAULT_SCHEMES:
        winner_advantage, loser_advantage = winner_loser_advantages(
            group_size,
            win_count,
            scheme,
            alpha=alpha,
        )
        print(
            f"  {_scheme_label(scheme, alpha):<22} "
            f"{winner_advantage:>10.3f}  {loser_advantage:>9.3f}"
        )
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare reward-only advantage normalization schemes for debate RL."
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="Number of rollouts per (question, side) group.",
    )
    parser.add_argument(
        "--ps",
        type=_parse_probabilities,
        default=DEFAULT_P_VALUES,
        help="Comma-separated win rates to evaluate, e.g. 0.1,0.2,0.5,0.8,0.9",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200_000,
        help="Monte Carlo trials per probability/scheme.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Monte Carlo simulation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Exponent for the exploratory power_mean scheme.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError(f"group_size must be positive, got {args.group_size}")

    print("Per-question weighting w(p) under reward-only schemes")
    print("Higher w(p) means the optimizer spends more update mass on that question/side.")
    print()

    for scheme in DEFAULT_SCHEMES:
        rows = simulate_scheme(
            args.ps,
            args.group_size,
            scheme=scheme,
            alpha=args.alpha,
            trials=args.trials,
            seed=args.seed,
        )
        _print_weight_table(scheme, rows, alpha=args.alpha)

    _print_fixed_group_example(group_size=8, win_count=6, alpha=args.alpha)


if __name__ == "__main__":
    main()
