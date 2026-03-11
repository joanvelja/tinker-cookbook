"""Batch-level gradient allocation experiment for debate advantage schemes.

This script samples question/side pairs with heterogeneous win rates ``p`` and
compares how the power-mean family

    A_i = (s_i - mean(s)) / (1 + mean(s)) ** alpha

allocates gradient mass across easy and hard sides, where ``s in {-1, +1}``.

Implementation note:
    The existing ``advantage_normalization_sim`` module works in ``{0, 1}``
    rewards. For fixed ``alpha`` the ``{-1, +1}`` formulation is exactly a
    constant rescaling of that scheme by ``2 ** (1 - alpha)``. Relative
    allocation across difficulty is therefore identical, while the reported
    absolute totals below match the user's settled ``{-1, +1}`` definition.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.advantage_normalization_sim import (
    exact_weight,
    winner_loser_advantages,
)

DEFAULT_ALPHAS = (0.0, 0.5, 1.0)
DEFAULT_BIN_EDGES = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)


@dataclass(frozen=True)
class BinSummary:
    label: str
    count: int
    mean_p: float
    mean_gradient: float
    total_gradient: float
    share_of_total: float


def _pm1_scale(alpha: float) -> float:
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    return 2.0 ** (1.0 - alpha)


def _pm1_exact_weight(p: float, group_size: int, alpha: float) -> float:
    if alpha == 0.0:
        base = exact_weight(p, group_size, scheme="mean_center")
    elif alpha == 1.0:
        base = exact_weight(p, group_size, scheme="maxrl")
    else:
        base = exact_weight(p, group_size, scheme="power_mean", alpha=alpha)
    return _pm1_scale(alpha) * base


def _pm1_winner_loser_advantages(group_size: int, win_count: int, alpha: float) -> tuple[float, float]:
    if alpha == 0.0:
        winner_advantage, loser_advantage = winner_loser_advantages(
            group_size,
            win_count,
            scheme="mean_center",
        )
    elif alpha == 1.0:
        winner_advantage, loser_advantage = winner_loser_advantages(
            group_size,
            win_count,
            scheme="maxrl",
        )
    else:
        winner_advantage, loser_advantage = winner_loser_advantages(
            group_size,
            win_count,
            scheme="power_mean",
            alpha=alpha,
        )
    scale = _pm1_scale(alpha)
    return scale * winner_advantage, scale * loser_advantage


def sample_question_win_rates(
    *,
    num_questions: int,
    p_min: float,
    p_max: float,
    seed: int,
) -> np.ndarray:
    if num_questions <= 0:
        raise ValueError(f"num_questions must be positive, got {num_questions}")
    if not 0.0 < p_min < p_max < 1.0:
        raise ValueError(f"Expected 0 < p_min < p_max < 1, got {p_min}, {p_max}")

    rng = np.random.default_rng(seed)
    return rng.uniform(p_min, p_max, size=num_questions)


def realized_question_gradients(
    p_values: np.ndarray,
    *,
    group_size: int,
    alpha: float,
    batch_repeats: int,
    seed: int,
) -> np.ndarray:
    if batch_repeats <= 0:
        raise ValueError(f"batch_repeats must be positive, got {batch_repeats}")

    rng = np.random.default_rng(seed)
    num_questions = p_values.shape[0]
    results = np.zeros((batch_repeats, num_questions), dtype=np.float64)

    for repeat_idx in range(batch_repeats):
        wins = rng.binomial(1, p_values[:, None], size=(num_questions, group_size)).astype(np.float64)
        win_counts = wins.sum(axis=1).astype(int)
        advantages = np.zeros_like(wins, dtype=np.float64)

        for win_count in range(group_size + 1):
            mask = win_counts == win_count
            if not np.any(mask):
                continue
            winner_advantage, loser_advantage = _pm1_winner_loser_advantages(
                group_size,
                win_count,
                alpha,
            )
            advantages[mask] = np.where(wins[mask] > 0, winner_advantage, loser_advantage)

        scores = np.where(wins > 0, 1.0 / p_values[:, None], -1.0 / (1.0 - p_values[:, None]))
        results[repeat_idx] = np.mean(advantages * scores, axis=1)

    return results


def summarize_bins(
    p_values: np.ndarray,
    gradients: np.ndarray,
    *,
    bin_edges: Sequence[float],
) -> list[BinSummary]:
    summaries: list[BinSummary] = []
    total_gradient = float(np.sum(gradients))

    for start, end in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if end == bin_edges[-1]:
            mask = (p_values >= start) & (p_values <= end)
            label = f"[{start:0.2f}, {end:0.2f}]"
        else:
            mask = (p_values >= start) & (p_values < end)
            label = f"[{start:0.2f}, {end:0.2f})"

        if not np.any(mask):
            summaries.append(
                BinSummary(
                    label=label,
                    count=0,
                    mean_p=float("nan"),
                    mean_gradient=0.0,
                    total_gradient=0.0,
                    share_of_total=0.0,
                )
            )
            continue

        bin_gradients = gradients[mask]
        total_bin_gradient = float(np.sum(bin_gradients))
        summaries.append(
            BinSummary(
                label=label,
                count=int(mask.sum()),
                mean_p=float(np.mean(p_values[mask])),
                mean_gradient=float(np.mean(bin_gradients)),
                total_gradient=total_bin_gradient,
                share_of_total=total_bin_gradient / total_gradient,
            )
        )

    return summaries


def _format_bin_table(title: str, summaries: Sequence[BinSummary]) -> str:
    lines = [title, "  p-bin          count  mean_p  mean_grad  total_grad  share"]
    for summary in summaries:
        lines.append(
            f"  {summary.label:<13} "
            f"{summary.count:>5d}  "
            f"{summary.mean_p:>6.3f}  "
            f"{summary.mean_gradient:>9.3f}  "
            f"{summary.total_gradient:>10.1f}  "
            f"{summary.share_of_total:>5.1%}"
        )
    return "\n".join(lines)


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _format_optional_float(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:0.3f}{suffix}"


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0.0}:
        return None
    return numerator / denominator


def run_experiment(
    *,
    num_questions: int,
    group_size: int,
    batch_repeats: int,
    p_min: float,
    p_max: float,
    seed: int,
    alphas: Sequence[float],
) -> None:
    p_values = sample_question_win_rates(
        num_questions=num_questions,
        p_min=p_min,
        p_max=p_max,
        seed=seed,
    )
    expected_by_alpha = {
        alpha: np.asarray([_pm1_exact_weight(p, group_size, alpha) for p in p_values], dtype=np.float64)
        for alpha in alphas
    }

    print("Question distribution")
    print(
        f"  num_questions={num_questions}  group_size={group_size}  "
        f"p_range=[{p_min:0.2f}, {p_max:0.2f}]  seed={seed}"
    )
    print(
        f"  p mean={np.mean(p_values):0.3f}  p median={np.median(p_values):0.3f}  "
        f"p min={np.min(p_values):0.3f}  p max={np.max(p_values):0.3f}"
    )
    print()

    baseline_total = float(np.sum(expected_by_alpha[0.0]))

    for alpha in alphas:
        expected = expected_by_alpha[alpha]
        realized = realized_question_gradients(
            p_values,
            group_size=group_size,
            alpha=alpha,
            batch_repeats=batch_repeats,
            seed=seed + int(alpha * 10_000) + 17,
        )
        realized_totals = np.sum(realized, axis=1)
        expected_total = float(np.sum(expected))
        easy_mask = p_values >= 0.8
        hard_mask = p_values < 0.2
        easy_mean = _safe_mean(expected[easy_mask])
        hard_mean = _safe_mean(expected[hard_mask])
        hard_share = float(np.sum(expected[hard_mask]) / expected_total) if np.any(hard_mask) else None
        easy_share = float(np.sum(expected[easy_mask]) / expected_total) if np.any(easy_mask) else None

        print(f"alpha={alpha:g}")
        print(
            f"  expected total gradient mass: {expected_total:0.1f}  "
            f"({expected_total / baseline_total:0.2f}x vs alpha=0)"
        )
        print(
            f"  realized total over {batch_repeats} rollout batches: "
            f"{np.mean(realized_totals):0.1f} +/- {np.std(realized_totals):0.1f}"
        )
        print(
            "  hard/easy mean question gradient: "
            f"{_format_optional_float(hard_mean)} / {_format_optional_float(easy_mean)} = "
            f"{_format_optional_float(_ratio_or_none(hard_mean, easy_mean), suffix='x')}"
        )
        print(
            "  hard/easy share of batch gradient: "
            f"{(f'{hard_share:0.1%}' if hard_share is not None else 'n/a')} / "
            f"{(f'{easy_share:0.1%}' if easy_share is not None else 'n/a')} = "
            f"{_format_optional_float(_ratio_or_none(hard_share, easy_share), suffix='x')}"
        )
        print(_format_bin_table("  expected gradient by difficulty bin", summarize_bins(p_values, expected, bin_edges=DEFAULT_BIN_EDGES)))
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate how power-mean debate advantages allocate gradient mass."
    )
    parser.add_argument("--num-questions", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--batch-repeats", type=int, default=256)
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--p-max", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,0.5,1",
        help="Comma-separated alpha values, e.g. 0,0.5,1",
    )
    return parser


def _parse_alphas(raw: str) -> tuple[float, ...]:
    alphas = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not alphas:
        raise ValueError("Expected at least one alpha value")
    for alpha in alphas:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
    return alphas


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_experiment(
        num_questions=args.num_questions,
        group_size=args.group_size,
        batch_repeats=args.batch_repeats,
        p_min=args.p_min,
        p_max=args.p_max,
        seed=args.seed,
        alphas=_parse_alphas(args.alphas),
    )


if __name__ == "__main__":
    main()
