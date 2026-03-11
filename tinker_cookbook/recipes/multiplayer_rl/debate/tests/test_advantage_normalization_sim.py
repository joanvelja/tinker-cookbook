from __future__ import annotations

import math

from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.advantage_normalization_sim import (
    exact_weight,
    rollout_advantages_for_win_count,
)


def test_reward_only_schemes_treat_all_winners_equally() -> None:
    advantages = rollout_advantages_for_win_count(
        group_size=8,
        win_count=6,
        scheme="maxrl",
    )

    assert advantages[:6] == [advantages[0]] * 6
    assert advantages[6:] == [advantages[6]] * 2


def test_exact_weight_matches_closed_forms() -> None:
    p = 0.2
    group_size = 16

    assert math.isclose(exact_weight(p, group_size, scheme="reinforce"), 1.0, rel_tol=1e-9)
    assert math.isclose(
        exact_weight(p, group_size, scheme="mean_center"),
        (group_size - 1) / group_size,
        rel_tol=1e-9,
    )
    assert math.isclose(
        exact_weight(p, group_size, scheme="maxrl"),
        (1.0 - (1.0 - p) ** (group_size - 1)) / p,
        rel_tol=1e-9,
    )


def test_power_mean_interpolates_between_centering_and_maxrl() -> None:
    p = 0.1
    group_size = 16

    mean_center = exact_weight(p, group_size, scheme="mean_center")
    tempered = exact_weight(p, group_size, scheme="power_mean", alpha=0.5)
    maxrl = exact_weight(p, group_size, scheme="maxrl")

    assert mean_center < tempered < maxrl
