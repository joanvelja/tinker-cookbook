"""Tests for rl/data_processing.py — advantage computation and normalization schemes."""

from typing import Sequence
from unittest.mock import MagicMock

import pytest
import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import (
    _normalize_subgroup,
    compute_advantages,
    remove_constant_reward_groups,
    trajectory_to_data,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    Env,
    Transition,
    TrajectoryGroup,
    Trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory_group(rewards: list[float]) -> TrajectoryGroup:
    """Build a minimal TrajectoryGroup whose get_total_rewards() returns *rewards*.

    Each trajectory has zero per-step reward so final_rewards_G drives total.
    """
    trajs = [Trajectory(transitions=[], final_ob=MagicMock()) for _ in rewards]
    return TrajectoryGroup(
        trajectories_G=trajs,
        final_rewards_G=rewards,
        metrics_G=[{} for _ in rewards],
    )


class _StubBuilder(EnvGroupBuilder):
    """Minimal builder that returns fixed subgroups."""

    def __init__(self, subgroups: tuple[tuple[int, ...], ...] | None = None):
        self._subgroups = subgroups

    async def make_envs(self) -> Sequence[Env]:
        raise NotImplementedError

    def advantage_subgroups(self, n_trajectories: int) -> tuple[tuple[int, ...], ...] | None:
        return self._subgroups


# ---------------------------------------------------------------------------
# 1. _normalize_subgroup: all schemes × both encodings
# ---------------------------------------------------------------------------


class TestNormalizeSubgroup:
    """Test _normalize_subgroup on known inputs for {0,1} and {-1,+1}."""

    # --- mean_center ---

    def test_mean_center_01(self):
        r = torch.tensor([1.0, 1.0, 0.0, 0.0])
        adv = _normalize_subgroup(r, "mean_center", alpha=0.0)
        expected = r - r.mean()
        assert torch.allclose(adv, expected)

    def test_mean_center_pm1(self):
        r = torch.tensor([1.0, 1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "mean_center", alpha=0.0)
        expected = r - r.mean()  # mean=0, so adv=r
        assert torch.allclose(adv, expected)

    # --- grpo ---

    def test_grpo_01(self):
        r = torch.tensor([1.0, 1.0, 0.0, 0.0])
        adv = _normalize_subgroup(r, "grpo", alpha=0.0)
        std = r.std()
        expected = (r - r.mean()) / std
        assert torch.allclose(adv, expected)

    def test_grpo_pm1(self):
        r = torch.tensor([1.0, 1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "grpo", alpha=0.0)
        std = r.std()
        expected = (r - r.mean()) / std
        assert torch.allclose(adv, expected)

    def test_grpo_constant_returns_zeros(self):
        r = torch.tensor([1.0, 1.0, 1.0])
        adv = _normalize_subgroup(r, "grpo", alpha=0.0)
        assert torch.allclose(adv, torch.zeros_like(r))

    # --- maxrl ---
    # New formula: p_eff = (mean - r_min) / (r_max - r_min), denom = p_eff^alpha
    # {0,1}: p_eff = mean, {-1,+1}: p_eff = (mean+1)/2

    def test_maxrl_01(self):
        # rewards {0,1}, p=0.5 → p_eff=0.5, denom=0.5
        r = torch.tensor([1.0, 1.0, 0.0, 0.0])
        adv = _normalize_subgroup(r, "maxrl", alpha=1.0)
        mean = r.mean()  # 0.5
        p_eff = mean  # {0,1}: p_eff = mean
        expected = (r - mean) / p_eff  # [0.5, 0.5, -0.5, -0.5] / 0.5 = [1, 1, -1, -1]
        assert torch.allclose(adv, expected)

    def test_maxrl_pm1(self):
        # rewards {-1,+1}, p=0.5 → mean=0, p_eff=(0+1)/2=0.5, denom=0.5
        r = torch.tensor([1.0, 1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "maxrl", alpha=1.0)
        mean = r.mean()  # 0.0
        p_eff = (mean - (-1.0)) / (1.0 - (-1.0))  # 0.5
        expected = (r - mean) / p_eff  # [1, 1, -1, -1] / 0.5 = [2, 2, -2, -2]
        assert torch.allclose(adv, expected)

    def test_maxrl_pm1_losing_side(self):
        # p=0.25: 1 win, 3 losses → mean=-0.5, p_eff=(-0.5+1)/2=0.25
        r = torch.tensor([1.0, -1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "maxrl", alpha=1.0)
        mean = r.mean()  # -0.5
        p_eff = (mean - (-1.0)) / (1.0 - (-1.0))  # 0.25
        expected = (r - mean) / p_eff  # [1.5, -0.5, -0.5, -0.5] / 0.25 = [6, -2, -2, -2]
        assert torch.allclose(adv, expected)
        # Winners get positive advantages, losers get negative — correct sign
        assert adv[0] > 0
        assert all(adv[i] < 0 for i in range(1, 4))

    # --- power_mean ---

    def test_power_mean_half_01(self):
        # {0,1}: p_eff = mean = 0.5, denom = 0.5^0.5
        r = torch.tensor([1.0, 1.0, 0.0, 0.0])
        adv = _normalize_subgroup(r, "power_mean", alpha=0.5)
        mean = r.mean()
        p_eff = mean  # 0.5
        denom = p_eff**0.5
        expected = (r - mean) / denom
        assert torch.allclose(adv, expected)

    def test_power_mean_half_pm1(self):
        # {-1,+1}: p_eff = (mean+1)/2 = 0.5, denom = 0.5^0.5
        r = torch.tensor([1.0, 1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "power_mean", alpha=0.5)
        mean = r.mean()  # 0.0
        p_eff = (mean - (-1.0)) / (1.0 - (-1.0))  # 0.5
        denom = p_eff**0.5
        expected = (r - mean) / denom
        assert torch.allclose(adv, expected)


# ---------------------------------------------------------------------------
# 2. Encoding-agnostic: MaxRL relative weighting preserved across encodings
# ---------------------------------------------------------------------------


class TestEncodingAgnostic:
    """MaxRL on {-1,+1} produces proportionally equivalent advantages as on {0,1}.

    "Proportional equivalence" means the ratio of winner advantage to loser
    advantage is the same, and the ordering is preserved.
    """

    @pytest.mark.parametrize("p_wins", [1, 2, 3])
    def test_maxrl_relative_ordering(self, p_wins: int):
        n = 4
        rewards_01 = torch.tensor([1.0] * p_wins + [0.0] * (n - p_wins))
        rewards_pm1 = torch.tensor([1.0] * p_wins + [-1.0] * (n - p_wins))

        adv_01 = _normalize_subgroup(rewards_01, "maxrl", alpha=1.0)
        adv_pm1 = _normalize_subgroup(rewards_pm1, "maxrl", alpha=1.0)

        # Both should have positive advantages for winners, negative for losers
        for i in range(n):
            if i < p_wins:
                assert adv_01[i] > 0
                assert adv_pm1[i] > 0
            else:
                assert adv_01[i] < 0
                assert adv_pm1[i] < 0

        # Ratio of winner to |loser| advantage should be the same
        ratio_01 = float(adv_01[0] / adv_01[-1].abs())
        ratio_pm1 = float(adv_pm1[0] / adv_pm1[-1].abs())
        assert abs(ratio_01 - ratio_pm1) < 1e-5, f"Ratios differ: {ratio_01} vs {ratio_pm1}"


# ---------------------------------------------------------------------------
# 3. Naive negative mean: (1+mean)^α stays positive
# ---------------------------------------------------------------------------


class TestNaiveNegativeMean:
    """With {-1,+1} rewards where mean < 0, verify p_eff-based denom stays positive."""

    def test_negative_mean_positive_denom(self):
        # p=0.25: 1 win, 3 losses → mean=-0.5, p_eff=(-0.5+1)/2=0.25
        r = torch.tensor([1.0, -1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "maxrl", alpha=1.0)

        # centered = [1.5, -0.5, -0.5, -0.5], denom = 0.25
        # adv = [6.0, -2.0, -2.0, -2.0]
        expected = torch.tensor([6.0, -2.0, -2.0, -2.0])
        assert torch.allclose(adv, expected)

        # Critically: winners get positive, losers get negative (no sign flip)
        assert adv[0] > 0
        assert all(adv[i] < 0 for i in range(1, 4))

    def test_power_mean_negative_mean(self):
        r = torch.tensor([1.0, -1.0, -1.0, -1.0])
        adv = _normalize_subgroup(r, "power_mean", alpha=0.5)
        mean = r.mean()  # -0.5
        p_eff = (mean - (-1.0)) / (1.0 - (-1.0))  # 0.25
        denom = p_eff**0.5  # 0.5
        expected = (r - mean) / denom
        assert torch.allclose(adv, expected)
        # Sign check
        assert adv[0] > 0
        assert all(adv[i] < 0 for i in range(1, 4))


# ---------------------------------------------------------------------------
# 4. Subgroup splitting
# ---------------------------------------------------------------------------


class TestSubgroupSplitting:
    """Mock TrajectoryGroup with interleaved A/B rewards. Splitting into subgroups
    gives different results than whole-group."""

    def test_interleaved_roles_different_from_whole_group(self):
        # Interleaved: A wins, B loses, A wins, B loses
        # A indices: [0, 2], B indices: [1, 3]
        rewards = [1.0, -1.0, 1.0, -1.0]  # {-1,+1} encoding
        tg = _make_trajectory_group(rewards)

        # Whole group: mean=0 → advantages = rewards (mean_center)
        whole_builder = _StubBuilder()
        [adv_whole] = compute_advantages([tg], [whole_builder])

        # Subgroups: A=[0,2], B=[1,3]
        sub_builder = _StubBuilder(subgroups=((0, 2), (1, 3)))
        [adv_sub] = compute_advantages([tg], [sub_builder])

        # Within subgroup A: rewards=[1,1], mean=1 → advantages=[0,0]
        # Within subgroup B: rewards=[-1,-1], mean=-1 → advantages=[0,0]
        # So with subgroups, all advantages are 0 (constant within each side)
        assert torch.allclose(adv_sub, torch.zeros(4))

        # But whole-group gives non-zero advantages (mean=0, adv=reward)
        assert not torch.allclose(adv_whole, torch.zeros(4))

    def test_asymmetric_subgroups(self):
        # A has mixed results, B always loses
        rewards = [1.0, -1.0, -1.0, -1.0]
        tg = _make_trajectory_group(rewards)

        sub_builder = _StubBuilder(subgroups=((0, 2), (1, 3)))
        [adv] = compute_advantages([tg], [sub_builder])

        # Subgroup A: rewards=[1, -1], mean=0 → adv=[1, -1]
        assert float(adv[0]) == pytest.approx(1.0)
        assert float(adv[2]) == pytest.approx(-1.0)

        # Subgroup B: rewards=[-1, -1], mean=-1 → adv=[0, 0]
        assert float(adv[1]) == pytest.approx(0.0)
        assert float(adv[3]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_win_01_zero(self):
        r = torch.tensor([1.0, 1.0, 1.0, 1.0])
        for scheme in ["mean_center", "grpo", "maxrl", "power_mean"]:
            adv = _normalize_subgroup(r, scheme, alpha=0.5)
            assert torch.allclose(adv, torch.zeros_like(r)), f"Failed for {scheme}"

    def test_all_lose_01_zero(self):
        r = torch.tensor([0.0, 0.0, 0.0, 0.0])
        for scheme in ["mean_center", "grpo", "maxrl", "power_mean"]:
            adv = _normalize_subgroup(r, scheme, alpha=0.5)
            assert torch.allclose(adv, torch.zeros_like(r)), f"Failed for {scheme}"

    def test_all_win_pm1_zero(self):
        r = torch.tensor([1.0, 1.0, 1.0, 1.0])
        for scheme in ["mean_center", "grpo", "maxrl", "power_mean"]:
            adv = _normalize_subgroup(r, scheme, alpha=0.5)
            assert torch.allclose(adv, torch.zeros_like(r)), f"Failed for {scheme}"

    def test_all_lose_pm1_zero(self):
        r = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        for scheme in ["mean_center", "grpo", "maxrl", "power_mean"]:
            adv = _normalize_subgroup(r, scheme, alpha=0.5)
            assert torch.allclose(adv, torch.zeros_like(r)), f"Failed for {scheme}"

    def test_single_element_zero(self):
        r = torch.tensor([1.0])
        for scheme in ["mean_center", "grpo", "maxrl", "power_mean"]:
            adv = _normalize_subgroup(r, scheme, alpha=0.5)
            assert torch.allclose(adv, torch.zeros_like(r)), f"Failed for {scheme}"

    def test_maxrl_equals_power_mean_alpha1(self):
        """maxrl is sugar for power_mean(alpha=1)."""
        for rewards in [
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, -1.0, -1.0]),
            torch.tensor([1.0, -1.0, -1.0, -1.0]),
        ]:
            adv_maxrl = _normalize_subgroup(rewards, "maxrl", alpha=1.0)
            adv_pm1 = _normalize_subgroup(rewards, "power_mean", alpha=1.0)
            assert torch.allclose(adv_maxrl, adv_pm1), (
                f"maxrl != power_mean(1) for rewards={rewards.tolist()}"
            )


# ---------------------------------------------------------------------------
# 6. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """compute_advantages with env_group_builders_P=None = old mean-centering."""

    def test_none_builders_mean_centers(self):
        rewards = [1.0, 1.0, 0.0, 0.0]
        tg = _make_trajectory_group(rewards)
        [adv] = compute_advantages([tg], env_group_builders_P=None)

        expected = torch.tensor(rewards) - torch.tensor(rewards).mean()
        assert torch.allclose(adv, expected)

    def test_none_builders_pm1(self):
        rewards = [1.0, 1.0, -1.0, -1.0]
        tg = _make_trajectory_group(rewards)
        [adv] = compute_advantages([tg], env_group_builders_P=None)

        expected = torch.tensor(rewards) - torch.tensor(rewards).mean()
        assert torch.allclose(adv, expected)

    def test_default_builder_matches_none(self):
        """Explicit default builder (no subgroups) gives same result as None builders."""
        rewards = [1.0, 0.0, 1.0, 0.0, 0.0]
        tg = _make_trajectory_group(rewards)

        [adv_none] = compute_advantages([tg], env_group_builders_P=None)

        default_builder = _StubBuilder()
        [adv_default] = compute_advantages([tg], [default_builder])

        assert torch.allclose(adv_none, adv_default)


# ---------------------------------------------------------------------------
# 7. remove_constant_reward_groups
# ---------------------------------------------------------------------------


class TestRemoveConstantRewardGroups:
    def test_removes_constant_groups(self):
        tg_mixed = _make_trajectory_group([1.0, 0.0])
        tg_all_win = _make_trajectory_group([1.0, 1.0])
        tg_all_lose = _make_trajectory_group([0.0, 0.0])

        result = remove_constant_reward_groups([tg_mixed, tg_all_win, tg_all_lose])
        assert len(result) == 1
        assert result[0] is tg_mixed

    def test_keeps_mixed_groups(self):
        tg1 = _make_trajectory_group([1.0, 0.0])
        tg2 = _make_trajectory_group([0.0, 1.0, 0.0])
        result = remove_constant_reward_groups([tg1, tg2])
        assert len(result) == 2

    def test_all_constant_returns_singleton(self):
        tg1 = _make_trajectory_group([1.0, 1.0])
        tg2 = _make_trajectory_group([0.0, 0.0])
        result = remove_constant_reward_groups([tg1, tg2])
        # Falls back to returning first group
        assert len(result) == 1
        assert result[0] is tg1


# ---------------------------------------------------------------------------
# 8. compute_advantages with schemes through the full pipeline
# ---------------------------------------------------------------------------


class TestComputeAdvantagesPipeline:
    """End-to-end tests going through compute_advantages with various configs."""

    def test_grpo_scheme(self):
        rewards = [1.0, 1.0, 0.0, 0.0]
        tg = _make_trajectory_group(rewards)
        [adv] = compute_advantages([tg], scheme="grpo")

        r = torch.tensor(rewards)
        expected = (r - r.mean()) / r.std()
        assert torch.allclose(adv, expected)

    def test_maxrl_scheme(self):
        rewards = [1.0, 1.0, 0.0, 0.0]
        tg = _make_trajectory_group(rewards)
        [adv] = compute_advantages([tg], scheme="maxrl")

        r = torch.tensor(rewards)
        mean = r.mean()  # 0.5
        p_eff = mean  # {0,1}: p_eff = mean
        expected = (r - mean) / p_eff
        assert torch.allclose(adv, expected)

    def test_multiple_groups_same_scheme(self):
        tg1 = _make_trajectory_group([1.0, 0.0])
        tg2 = _make_trajectory_group([1.0, 1.0, 0.0])

        advs = compute_advantages([tg1, tg2], scheme="maxrl")
        assert len(advs) == 2

        # Group 1: maxrl on [1, 0] → mean=0.5, p_eff=0.5 → [1.0, -1.0]
        assert torch.allclose(advs[0], torch.tensor([1.0, -1.0]))

        # Group 2: maxrl on [1, 1, 0] → mean=2/3, p_eff=2/3
        r2 = torch.tensor([1.0, 1.0, 0.0])
        mean2 = r2.mean()
        expected2 = (r2 - mean2) / mean2  # {0,1}: p_eff = mean
        assert torch.allclose(advs[1], expected2)

    def test_subgroups_with_maxrl(self):
        # Two roles interleaved: A=[0,2], B=[1,3]
        # A: rewards [1, 0], B: rewards [0, 1]
        rewards = [1.0, 0.0, 0.0, 1.0]
        tg = _make_trajectory_group(rewards)

        builder = _StubBuilder(subgroups=((0, 2), (1, 3)))
        [adv] = compute_advantages([tg], [builder], scheme="maxrl")

        # Each subgroup {0,1}: mean=0.5, p_eff=0.5
        # centered = [0.5, -0.5], divided by 0.5 → [1.0, -1.0]
        assert float(adv[0]) == pytest.approx(1.0)  # A winner
        assert float(adv[2]) == pytest.approx(-1.0)  # A loser
        assert float(adv[1]) == pytest.approx(-1.0)  # B loser
        assert float(adv[3]) == pytest.approx(1.0)  # B winner


# ---------------------------------------------------------------------------
# 9. Cross-validation against advantage_normalization_sim.py
# ---------------------------------------------------------------------------


class TestSimCrossValidation:
    """Verify _normalize_subgroup matches the sim script's winner_loser_advantages.

    Both use p^α as denominator for {0,1} rewards, so mean_center/maxrl/power_mean
    match exactly. GRPO differs: the sim uses population std (sqrt(p*(1-p))) while
    the code uses sample std (Bessel correction). They differ by sqrt(n/(n-1)) but
    produce the same winner/loser ratio.
    """

    @pytest.mark.parametrize(
        "group_size,win_count",
        [(4, 1), (4, 2), (4, 3), (8, 2), (8, 5), (8, 7)],
    )
    @pytest.mark.parametrize("scheme", ["mean_center", "maxrl", "power_mean"])
    def test_exact_match_with_sim(self, group_size, win_count, scheme):
        from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.advantage_normalization_sim import (
            winner_loser_advantages,
        )

        alpha = 0.5
        sim_winner, sim_loser = winner_loser_advantages(group_size, win_count, scheme, alpha=alpha)

        rewards = torch.tensor([1.0] * win_count + [0.0] * (group_size - win_count))
        adv = _normalize_subgroup(rewards, scheme, alpha=alpha)
        code_winner = float(adv[0])
        code_loser = float(adv[-1])

        assert code_winner == pytest.approx(sim_winner, abs=1e-6), (
            f"Winner mismatch for {scheme} g={group_size} k={win_count}: "
            f"sim={sim_winner:.6f} code={code_winner:.6f}"
        )
        assert code_loser == pytest.approx(sim_loser, abs=1e-6), (
            f"Loser mismatch for {scheme} g={group_size} k={win_count}: "
            f"sim={sim_loser:.6f} code={code_loser:.6f}"
        )

    @pytest.mark.parametrize(
        "group_size,win_count",
        [(4, 1), (4, 2), (4, 3), (8, 2), (8, 5), (8, 7)],
    )
    def test_grpo_ratio_matches_sim(self, group_size, win_count):
        """GRPO: sim uses population std, code uses sample std. Ratio still matches."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.advantage_normalization_sim import (
            winner_loser_advantages,
        )

        sim_winner, sim_loser = winner_loser_advantages(group_size, win_count, "grpo", alpha=0.5)

        rewards = torch.tensor([1.0] * win_count + [0.0] * (group_size - win_count))
        adv = _normalize_subgroup(rewards, "grpo", alpha=0.5)
        code_winner = float(adv[0])
        code_loser = float(adv[-1])

        if sim_winner == 0.0:
            assert code_winner == pytest.approx(0.0, abs=1e-7)
            return

        # Winner/loser ratio is preserved (std is a single scalar)
        sim_ratio = sim_winner / sim_loser
        code_ratio = code_winner / code_loser
        assert sim_ratio == pytest.approx(code_ratio, rel=1e-5)


# ---------------------------------------------------------------------------
# 10. Per-trajectory-mean advantage normalization
# ---------------------------------------------------------------------------


def _make_transition(n_action_tokens: int, n_ob_tokens: int = 3) -> Transition:
    """Build a minimal Transition with *n_action_tokens* action tokens."""
    ob = tinker.ModelInput.from_ints(list(range(100, 100 + n_ob_tokens)))
    ac = TokensWithLogprobs(
        tokens=list(range(200, 200 + n_action_tokens)),
        maybe_logprobs=[-0.5] * n_action_tokens,
    )
    return Transition(ob=ob, ac=ac, reward=0.0, episode_done=False)


def _make_single_turn_trajectory(n_action_tokens: int) -> Trajectory:
    """Single-turn trajectory with *n_action_tokens* action tokens."""
    transition = _make_transition(n_action_tokens, n_ob_tokens=3)
    final_ob = tinker.ModelInput.from_ints([999])
    return Trajectory(transitions=[transition], final_ob=final_ob)


def _extract_advantages(data: list[tinker.Datum]) -> torch.Tensor:
    """Extract the advantages tensor from trajectory_to_data output (concatenated)."""
    parts = []
    for datum in data:
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        parts.append(adv)
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _extract_mask(data: list[tinker.Datum]) -> torch.Tensor:
    parts = []
    for datum in data:
        m = datum.loss_fn_inputs["mask"].to_torch()
        parts.append(m)
    return torch.cat(parts) if len(parts) > 1 else parts[0]


class TestNormalizeAdvantagesByLength:
    """Test per-trajectory-mean advantage normalization (normalize_advantages_by_length)."""

    def test_unnormalized_broadcasts_scalar(self):
        """Without normalization, all action tokens get the raw advantage."""
        traj = _make_single_turn_trajectory(n_action_tokens=10)
        data = trajectory_to_data(traj, traj_advantage=2.0, normalize_advantages_by_length=False)
        adv = _extract_advantages(data)
        mask = _extract_mask(data)
        action_advs = adv[mask > 0]
        assert all(a == pytest.approx(2.0) for a in action_advs.tolist())

    def test_normalized_divides_by_action_count(self):
        """With normalization, each action token gets advantage / n_action_tokens."""
        n_tokens = 10
        traj = _make_single_turn_trajectory(n_action_tokens=n_tokens)
        data = trajectory_to_data(traj, traj_advantage=2.0, normalize_advantages_by_length=True)
        adv = _extract_advantages(data)
        mask = _extract_mask(data)
        action_advs = adv[mask > 0]
        expected_per_token = 2.0 / n_tokens
        assert all(a == pytest.approx(expected_per_token) for a in action_advs.tolist())

    def test_sum_of_advantages_equals_original(self):
        """The sum over action tokens should equal the original advantage (length cancels)."""
        n_tokens = 10
        raw_advantage = 3.5
        traj = _make_single_turn_trajectory(n_action_tokens=n_tokens)
        data = trajectory_to_data(traj, traj_advantage=raw_advantage, normalize_advantages_by_length=True)
        adv = _extract_advantages(data)
        mask = _extract_mask(data)
        action_advs = adv[mask > 0]
        assert float(action_advs.sum()) == pytest.approx(raw_advantage)

    def test_multi_transition_trajectory(self):
        """Multiple transitions: advantage divided by total action tokens across all."""
        # Build a 2-transition trajectory where observations are prefixes
        ob1 = tinker.ModelInput.from_ints([10, 11, 12])
        ac1 = TokensWithLogprobs(tokens=[20, 21], maybe_logprobs=[-0.5, -0.5])  # 2 tokens
        t1 = Transition(ob=ob1, ac=ac1, reward=0.0, episode_done=False)

        # ob2 is a prefix extension: ob1 + ac1 + new_obs
        ob2 = tinker.ModelInput.from_ints([10, 11, 12, 20, 21, 30, 31])
        ac2 = TokensWithLogprobs(tokens=[40, 41, 42], maybe_logprobs=[-0.5, -0.5, -0.5])  # 3 tokens
        t2 = Transition(ob=ob2, ac=ac2, reward=0.0, episode_done=True)

        traj = Trajectory(transitions=[t1, t2], final_ob=tinker.ModelInput.from_ints([999]))

        total_action_tokens = 2 + 3  # = 5
        raw_advantage = 5.0

        data_norm = trajectory_to_data(traj, traj_advantage=raw_advantage, normalize_advantages_by_length=True)
        data_raw = trajectory_to_data(traj, traj_advantage=raw_advantage, normalize_advantages_by_length=False)

        adv_norm = _extract_advantages(data_norm)
        mask_norm = _extract_mask(data_norm)
        action_advs_norm = adv_norm[mask_norm > 0]

        adv_raw = _extract_advantages(data_raw)
        mask_raw = _extract_mask(data_raw)
        action_advs_raw = adv_raw[mask_raw > 0]

        # Normalized: each token gets 5.0 / 5 = 1.0
        expected_per_token = raw_advantage / total_action_tokens
        assert all(a == pytest.approx(expected_per_token) for a in action_advs_norm.tolist())

        # Unnormalized: each token gets 5.0
        assert all(a == pytest.approx(raw_advantage) for a in action_advs_raw.tolist())

        # Sum of normalized advantages = original advantage
        assert float(action_advs_norm.sum()) == pytest.approx(raw_advantage)

    def test_different_lengths_contribute_equally(self):
        """Two trajectories of different lengths get equal total gradient contribution."""
        short_traj = _make_single_turn_trajectory(n_action_tokens=5)
        long_traj = _make_single_turn_trajectory(n_action_tokens=50)
        advantage = 1.0

        data_short = trajectory_to_data(short_traj, traj_advantage=advantage, normalize_advantages_by_length=True)
        data_long = trajectory_to_data(long_traj, traj_advantage=advantage, normalize_advantages_by_length=True)

        adv_short = _extract_advantages(data_short)
        mask_short = _extract_mask(data_short)
        adv_long = _extract_advantages(data_long)
        mask_long = _extract_mask(data_long)

        sum_short = float((adv_short * mask_short).sum())
        sum_long = float((adv_long * mask_long).sum())

        # Both should sum to the same value (the raw advantage)
        assert sum_short == pytest.approx(sum_long)
        assert sum_short == pytest.approx(advantage)

    def test_observation_tokens_always_zero(self):
        """Observation tokens should have zero advantage regardless of normalization."""
        traj = _make_single_turn_trajectory(n_action_tokens=10)
        for normalize in [True, False]:
            data = trajectory_to_data(traj, traj_advantage=2.0, normalize_advantages_by_length=normalize)
            adv = _extract_advantages(data)
            mask = _extract_mask(data)
            ob_advs = adv[mask == 0]
            assert all(a == pytest.approx(0.0) for a in ob_advs.tolist())


# ---------------------------------------------------------------------------
# 11. Exclude mechanism (judge failures)
# ---------------------------------------------------------------------------


def _make_trajectory_group_with_exclusions(
    rewards: list[float], excluded: list[bool]
) -> TrajectoryGroup:
    """Build a TrajectoryGroup where some trajectories are excluded.

    Each trajectory has a single transition with exclude set per *excluded*.
    """
    assert len(rewards) == len(excluded)
    trajs = []
    for r, exc in zip(rewards, excluded):
        t = _make_transition(n_action_tokens=5)
        t.exclude = exc
        t.reward = r
        traj = Trajectory(transitions=[t], final_ob=MagicMock())
        trajs.append(traj)
    return TrajectoryGroup(
        trajectories_G=trajs,
        final_rewards_G=[0.0] * len(rewards),  # step rewards only
        metrics_G=[{} for _ in rewards],
    )


class TestTrajectoryIsExcluded:
    def test_not_excluded(self):
        traj = _make_single_turn_trajectory(n_action_tokens=5)
        assert not traj.is_excluded

    def test_excluded(self):
        t = _make_transition(n_action_tokens=5)
        t.exclude = True
        traj = Trajectory(transitions=[t], final_ob=MagicMock())
        assert traj.is_excluded


class TestExcludeMechanism:
    """Verify that excluded trajectories are filtered before advantage computation."""

    def test_some_excluded_only_valid_get_advantages(self):
        """Group with some excluded trajectories -> only valid ones remain."""
        # 4 trajectories: indices 1 and 3 are excluded
        rewards = [1.0, 0.5, 0.0, 0.5]
        excluded = [False, True, False, True]
        tg = _make_trajectory_group_with_exclusions(rewards, excluded)

        # Filter manually (simulating what prepare_minibatch does)
        valid_indices = [i for i, t in enumerate(tg.trajectories_G) if not t.is_excluded]
        assert valid_indices == [0, 2]

        filtered_tg = TrajectoryGroup(
            trajectories_G=[tg.trajectories_G[i] for i in valid_indices],
            final_rewards_G=[tg.final_rewards_G[i] for i in valid_indices],
            metrics_G=[tg.metrics_G[i] for i in valid_indices],
        )

        # Advantages should be computed on the two valid trajectories only
        [adv] = compute_advantages([filtered_tg])
        assert len(adv) == 2
        # rewards are [1.0, 0.0], mean=0.5 -> advantages [0.5, -0.5]
        assert float(adv[0]) == pytest.approx(0.5)
        assert float(adv[1]) == pytest.approx(-0.5)

    def test_all_excluded_drops_group(self):
        """Group with all excluded trajectories -> dropped entirely."""
        rewards = [1.0, 0.0, 0.5]
        excluded = [True, True, True]
        tg = _make_trajectory_group_with_exclusions(rewards, excluded)

        valid_indices = [i for i, t in enumerate(tg.trajectories_G) if not t.is_excluded]
        assert len(valid_indices) == 0

    def test_one_valid_drops_group(self):
        """Group with only 1 valid trajectory -> dropped (can't center advantages)."""
        rewards = [1.0, 0.0, 0.5]
        excluded = [True, True, False]
        tg = _make_trajectory_group_with_exclusions(rewards, excluded)

        valid_indices = [i for i, t in enumerate(tg.trajectories_G) if not t.is_excluded]
        assert len(valid_indices) == 1
        # < 2 valid → group should be dropped

    def test_no_exclusions_unchanged(self):
        """No exclusions -> behavior is identical to before."""
        rewards = [1.0, 0.0, 1.0, 0.0]
        excluded = [False, False, False, False]
        tg = _make_trajectory_group_with_exclusions(rewards, excluded)

        valid_indices = [i for i, t in enumerate(tg.trajectories_G) if not t.is_excluded]
        assert len(valid_indices) == 4

        # Advantages unchanged
        [adv] = compute_advantages([tg])
        expected = torch.tensor(rewards) - torch.tensor(rewards).mean()
        assert torch.allclose(adv, expected)
