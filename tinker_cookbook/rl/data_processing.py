"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging
from typing import List, Sequence

import tinker
import torch
from torch import Tensor
from tinker import TensorData
from tinker_cookbook.rl.types import AdvantageScheme, EnvGroupBuilder, Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def _normalize_subgroup(rewards: Tensor, scheme: AdvantageScheme, alpha: float) -> Tensor:
    """Normalize advantages within a subgroup using the specified scheme.

    For power_mean/maxrl, computes effective win rate p_eff = (mean - r_min) / (r_max - r_min)
    and divides centered rewards by p_eff^alpha. This is encoding-agnostic:
    - {0,1} rewards: p_eff = mean = p, denom = p^alpha (matches MaxRL paper exactly)
    - {-1,+1} rewards: p_eff = (mean+1)/2 = p, denom = p^alpha (same weighting)
    """
    if len(rewards) < 2:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    centered = rewards - mean

    if scheme == "mean_center":
        return centered

    if scheme == "grpo":
        std = rewards.std()
        return centered / std if std > 0 else torch.zeros_like(rewards)

    if scheme not in ("maxrl", "power_mean"):
        raise ValueError(f"Unknown advantage scheme: {scheme!r}")

    # power_mean family (includes maxrl as alpha=1)
    eff_alpha = alpha if scheme == "power_mean" else 1.0
    r_min = rewards.min()
    r_max = rewards.max()
    r_range = r_max - r_min
    if r_range < 1e-8:
        return torch.zeros_like(rewards)  # constant rewards
    p_eff = (mean - r_min) / r_range
    if p_eff <= 1e-8:
        return torch.zeros_like(rewards)  # all-lose
    return centered / p_eff**eff_alpha


def compute_advantages(
    trajectory_groups_P: List[TrajectoryGroup],
    env_group_builders_P: Sequence[EnvGroupBuilder] | None = None,
    *,
    scheme: AdvantageScheme = "mean_center",
    alpha: float = 0.5,
) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups or subgroups.

    Args:
        trajectory_groups_P: Trajectory groups to compute advantages for.
        env_group_builders_P: Builders providing subgroup partitions (structural).
            None = single group per trajectory group.
        scheme: Normalization scheme (training-level objective choice).
        alpha: Exponent for power_mean. maxrl ignores this (uses α=1).
    """
    advantages_P: list[torch.Tensor] = []

    for i, traj_group in enumerate(trajectory_groups_P):
        rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float32)
        n = len(rewards_G)

        subgroups = (
            env_group_builders_P[i].advantage_subgroups(n)
            if env_group_builders_P is not None
            else None
        )

        if subgroups is None:
            advantages_G = _normalize_subgroup(rewards_G, scheme, alpha)
        else:
            advantages_G = torch.zeros_like(rewards_G)
            for sub_indices in subgroups:
                idx = torch.tensor(sub_indices, dtype=torch.long)
                sub_rewards = rewards_G[idx]
                sub_adv = _normalize_subgroup(sub_rewards, scheme, alpha)
                advantages_G[idx] = sub_adv

        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(
    traj: Trajectory,
    traj_advantage: float,
    *,
    normalize_advantages_by_length: bool = False,
) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    # When normalizing by length, divide the trajectory advantage by the total
    # number of action tokens so each trajectory contributes equally to the
    # gradient regardless of sequence length.
    if normalize_advantages_by_length:
        total_action_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
        per_token_advantage = traj_advantage / total_action_tokens if total_action_tokens > 0 else 0.0
    else:
        per_token_advantage = traj_advantage

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [per_token_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
    *,
    normalize_advantages_by_length: bool = False,
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(
                traj,
                float(traj_advantage),
                normalize_advantages_by_length=normalize_advantages_by_length,
            )
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
