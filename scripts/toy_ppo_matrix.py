"""
Toy PPO/CISPO/REINFORCE matrix experiment.

12 configs: {token_sum, per_traj_mean} x {maxrl, mean_center} x {PPO(0.8/1.28), CISPO(0.8/1.28), REINFORCE}
3 seeds each = 36 runs.

Model: 2-layer MLP, hidden=256, vocab=100.
LR=1e-3, no grad clipping. 500 steps. B=16 groups, G=8 rollouts/group.

Implementation notes:
- Preserves the current toy data generation exactly: random variable-length sequences,
  random next-token targets, and the real `_normalize_subgroup` semantics.
- Avoids per-token MLP work by aggregating the batch into weighted
  `(input_token, target_token)` tables. Because the toy MLP is token-local, the
  logits depend only on the current token id.
- PPO/CISPO/REINFORCE share the same gradient path here because `old_logprobs`
  are detached current-policy outputs, so the script only trains 12 unique
  trajectories and fans results back out to 36 reported configs.
"""

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the real advantage normalization
sys.path.insert(0, "/Users/joalja/Documents/Github/ext/tinker-cookbook")
from tinker_cookbook.rl.data_processing import _normalize_subgroup


DEFAULT_MAX_TOK = 200
METRIC_INTERVAL = 10
GRAD_DECOMP_STEPS = frozenset({0, 250, 499})
CATEGORY_NAMES = ("correct", "wrong", "truncated")


# ─── Reward distribution (from real GPQA data) ─────────────────────────────────

def sample_episode(rng: np.random.Generator):
    """Sample a single (reward, length, category) tuple from the empirical toy mix."""
    u = rng.random()
    if u < 0.37:  # correct
        length = max(1, int(rng.normal(150, 40)))
        return 1.0, length, "correct"
    if u < 0.70:  # wrong
        length = max(1, int(rng.normal(180, 50)))
        return 0.0, length, "wrong"
    return -0.2, 410, "truncated"


def sample_batch(
    np_rng: np.random.Generator,
    batch_size: int,
    max_tok: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized version of `sample_episode` for one PPO batch."""
    u = np_rng.random(batch_size)

    rewards = np.empty(batch_size, dtype=np.float32)
    lengths = np.empty(batch_size, dtype=np.int64)
    categories = np.empty(batch_size, dtype=np.int8)

    correct = u < 0.37
    wrong = (u >= 0.37) & (u < 0.70)
    truncated = ~(correct | wrong)

    rewards[correct] = 1.0
    rewards[wrong] = 0.0
    rewards[truncated] = -0.2

    categories[correct] = 0
    categories[wrong] = 1
    categories[truncated] = 2

    if correct.any():
        lengths[correct] = np.maximum(1, np_rng.normal(150, 40, size=correct.sum()).astype(np.int64))
    if wrong.any():
        lengths[wrong] = np.maximum(1, np_rng.normal(180, 50, size=wrong.sum()).astype(np.int64))
    if truncated.any():
        lengths[truncated] = 410

    return rewards, np.minimum(lengths, max_tok), categories


def normalize_advantages_batch(
    rewards_BG: torch.Tensor,
    scheme: str,
    alpha: float,
) -> torch.Tensor:
    """Batched equivalent of `_normalize_subgroup` for fixed-size groups."""
    if rewards_BG.shape[1] < 2:
        return torch.zeros_like(rewards_BG)

    mean = rewards_BG.mean(dim=1, keepdim=True)
    centered = rewards_BG - mean

    if scheme == "mean_center":
        return centered

    if scheme == "grpo":
        std = rewards_BG.std(dim=1, keepdim=True)
        return torch.where(std > 0, centered / std, torch.zeros_like(centered))

    if scheme not in ("maxrl", "power_mean"):
        return torch.stack([_normalize_subgroup(row, scheme, alpha) for row in rewards_BG])

    eff_alpha = alpha if scheme == "power_mean" else 1.0
    r_min = rewards_BG.min(dim=1, keepdim=True).values
    r_max = rewards_BG.max(dim=1, keepdim=True).values
    r_range = r_max - r_min
    p_eff = (mean - r_min) / r_range

    valid = (r_range >= 1e-8) & (p_eff > 1e-8)
    denom = torch.where(valid, p_eff.pow(eff_alpha), torch.ones_like(p_eff))
    return torch.where(valid, centered / denom, torch.zeros_like(centered))


# ─── Model ──────────────────────────────────────────────────────────────────────

class ToyPolicy(nn.Module):
    def __init__(self, vocab_size=100, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (...,) -> logits: (..., V)"""
        x = self.embed(tokens)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def input_log_probs(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Return log-prob tables for a batch of input token ids."""
        return F.log_softmax(self.forward(input_tokens), dim=-1)


# ─── Loss functions ─────────────────────────────────────────────────────────────

LossType = Literal["ppo", "cispo", "reinforce"]


def compute_aggregated_loss(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    positive_advantages: torch.Tensor,
    negative_advantages: torch.Tensor,
    loss_type: LossType,
    clip_lo: float = 0.8,
    clip_hi: float = 1.28,
) -> torch.Tensor:
    """Compute the loss from aggregated per-(input,target) advantage weights."""
    if loss_type == "reinforce":
        signed_advantages = positive_advantages - negative_advantages
        return -(signed_advantages * new_logprobs).sum()

    ratio = torch.exp(new_logprobs - old_logprobs)

    if loss_type == "ppo":
        clipped = torch.clamp(ratio, clip_lo, clip_hi)
        positive_term = torch.minimum(ratio, clipped)
        negative_term = torch.maximum(ratio, clipped)
        return -(positive_advantages * positive_term).sum() + (negative_advantages * negative_term).sum()

    if loss_type == "cispo":
        positive_term = torch.clamp(ratio, max=clip_hi)
        negative_term = torch.clamp(ratio, min=clip_lo)
        return -(positive_advantages * positive_term).sum() + (negative_advantages * negative_term).sum()

    raise ValueError(f"Unknown loss type: {loss_type}")


# ─── Reference sequence for correctness proxy ───────────────────────────────────

def make_reference_sequence(vocab_size=100, length=10, seed=42):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.integers(0, vocab_size, size=(1, length)), dtype=torch.long)


# ─── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    loss_agg: str        # "token_sum" or "per_traj_mean"
    adv_scheme: str      # "maxrl" or "mean_center"
    loss_type: LossType  # "ppo", "cispo", "reinforce"
    seed: int

    @property
    def name(self) -> str:
        return f"{self.loss_agg}_{self.adv_scheme}_{self.loss_type}"


def build_configs() -> list[Config]:
    configs: list[Config] = []
    loss_aggs = ["token_sum", "per_traj_mean"]
    adv_schemes = ["maxrl", "mean_center"]
    loss_types: list[LossType] = ["ppo", "cispo", "reinforce"]
    seeds = [0, 1, 2]

    for loss_agg, adv_scheme, loss_type in itertools.product(loss_aggs, adv_schemes, loss_types):
        for seed in seeds:
            configs.append(
                Config(
                    loss_agg=loss_agg,
                    adv_scheme=adv_scheme,
                    loss_type=loss_type,
                    seed=seed,
                )
            )
    return configs


def build_transition_tables(
    rewards: np.ndarray,
    lengths: np.ndarray,
    category_codes: np.ndarray,
    cfg: Config,
    *,
    B: int,
    G: int,
    vocab_size: int,
    token_buffer: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build aggregated transition tables for the current random-sequence batch."""
    rewards_BG = torch.from_numpy(rewards.reshape(B, G))
    trajectory_advantages = normalize_advantages_batch(rewards_BG, cfg.adv_scheme, alpha=1.0).reshape(-1)

    action_lengths = torch.from_numpy(np.clip(lengths - 1, a_min=0, a_max=None).astype(np.int64, copy=False))
    per_token_advantages = trajectory_advantages
    if cfg.loss_agg == "per_traj_mean":
        per_token_advantages = trajectory_advantages / action_lengths.clamp(min=1).to(torch.float32)

    token_buffer.random_(0, vocab_size)
    valid_mask = position_ids.unsqueeze(0) < action_lengths.unsqueeze(1)
    input_tokens = token_buffer[:, :-1][valid_mask]
    target_tokens = token_buffer[:, 1:][valid_mask]
    pair_ids = input_tokens * vocab_size + target_tokens

    trajectory_categories = torch.from_numpy(category_codes.astype(np.int64, copy=False))
    episode_indices = torch.repeat_interleave(torch.arange(B * G, dtype=torch.int64), action_lengths)
    advantages_flat = per_token_advantages[episode_indices]
    category_codes_flat = trajectory_categories[episode_indices]

    n_pair_buckets = vocab_size * vocab_size
    positive_weights = torch.bincount(
        pair_ids,
        weights=advantages_flat.clamp_min(0),
        minlength=n_pair_buckets,
    ).reshape(vocab_size, vocab_size)
    negative_weights = torch.bincount(
        pair_ids,
        weights=(-advantages_flat.clamp_max(0)),
        minlength=n_pair_buckets,
    ).reshape(vocab_size, vocab_size)

    category_weights = torch.zeros(len(CATEGORY_NAMES), 2, vocab_size, vocab_size, dtype=torch.float32)
    for code in range(len(CATEGORY_NAMES)):
        category_mask = category_codes_flat == code
        if bool(category_mask.any().item()):
            category_pair_ids = pair_ids[category_mask]
            category_advantages = advantages_flat[category_mask]
            category_weights[code, 0] = torch.bincount(
                category_pair_ids,
                weights=category_advantages.clamp_min(0),
                minlength=n_pair_buckets,
            ).reshape(vocab_size, vocab_size)
            category_weights[code, 1] = torch.bincount(
                category_pair_ids,
                weights=(-category_advantages.clamp_max(0)),
                minlength=n_pair_buckets,
            ).reshape(vocab_size, vocab_size)

    if cfg.loss_agg == "token_sum":
        per_traj_budget = per_token_advantages.abs() * action_lengths.to(torch.float32)
    else:
        per_traj_budget = per_token_advantages.abs() * (action_lengths > 0).to(torch.float32)

    budget_by_category = torch.zeros(len(CATEGORY_NAMES), dtype=torch.float32)
    budget_by_category.index_add_(0, trajectory_categories, per_traj_budget)

    pair_weights = torch.stack((positive_weights, negative_weights))
    return pair_weights, category_weights, budget_by_category


def flatten_current_grads(model: nn.Module) -> torch.Tensor:
    flat_grads = []
    for param in model.parameters():
        if param.grad is None:
            flat_grads.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))
        else:
            flat_grads.append(param.grad.detach().flatten())
    return torch.cat(flat_grads)


# ─── Main training loop ─────────────────────────────────────────────────────────

def run_experiment(
    cfg: Config,
    n_steps: int = 500,
    B: int = 16,
    G: int = 8,
    vocab_size: int = 100,
    hidden: int = 256,
    max_tok: int = DEFAULT_MAX_TOK,
):
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    model = ToyPolicy(vocab_size, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ref_seq = make_reference_sequence(vocab_size)
    ref_input_tokens = ref_seq[:, :-1].reshape(-1)
    ref_target_tokens = ref_seq[:, 1:].reshape(-1)
    vocab_tokens = torch.arange(vocab_size, dtype=torch.long)

    batch_size = B * G
    position_ids = torch.arange(max_tok - 1)
    token_buffer = torch.empty(batch_size, max_tok, dtype=torch.long)

    metrics_log = []
    grad_decomp_log = {}

    for step in range(n_steps):
        rewards, lengths, category_codes = sample_batch(np_rng, batch_size, max_tok)
        pair_weights, category_weights, budget_by_category = build_transition_tables(
            rewards,
            lengths,
            category_codes,
            cfg,
            B=B,
            G=G,
            vocab_size=vocab_size,
            token_buffer=token_buffer,
            position_ids=position_ids,
        )

        model.train()
        new_logprobs = model.input_log_probs(vocab_tokens)
        old_logprobs = new_logprobs.detach()

        loss = compute_aggregated_loss(
            old_logprobs,
            new_logprobs,
            pair_weights[0],
            pair_weights[1],
            cfg.loss_type,
        )

        need_decomp = cfg.seed == 0 and step in GRAD_DECOMP_STEPS
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=need_decomp)

        if need_decomp:
            grad_decomp_log[step] = gradient_decomposition(
                model,
                old_logprobs,
                new_logprobs,
                category_weights,
                budget_by_category,
                cfg,
            )

        if step % METRIC_INTERVAL == 0:
            total_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm_sq += param.grad.data.norm(2).item() ** 2
            grad_norm = total_norm_sq**0.5

            with torch.no_grad():
                log_probs = new_logprobs.detach()
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(-1).mean().item()
                ref_lp = log_probs[ref_input_tokens, ref_target_tokens].sum().item()
                loss_by_type = {
                    loss_type: compute_aggregated_loss(
                        old_logprobs,
                        new_logprobs,
                        pair_weights[0],
                        pair_weights[1],
                        loss_type,
                    ).item()
                    for loss_type in ("ppo", "cispo", "reinforce")
                }

            metrics_log.append(
                {
                    "step": step,
                    "entropy": entropy,
                    "grad_norm": grad_norm,
                    "ref_logprob": ref_lp,
                    "losses": loss_by_type,
                }
            )

        if not torch.isfinite(loss):
            metrics_log.append(
                {
                    "step": step,
                    "entropy": float("nan"),
                    "grad_norm": float("inf"),
                    "ref_logprob": float("nan"),
                    "losses": {"ppo": float("nan"), "cispo": float("nan"), "reinforce": float("nan")},
                }
            )
            break

        optimizer.step()

    return metrics_log, grad_decomp_log


def gradient_decomposition(
    model: nn.Module,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    category_weights: torch.Tensor,
    budget_by_category: torch.Tensor,
    cfg: Config,
):
    """Decompose gradient by category: truncated, wrong, correct."""
    params = tuple(model.parameters())
    total_grads = [None if param.grad is None else param.grad.detach().clone() for param in params]
    total_grad_flat = flatten_current_grads(model)
    total_norm = total_grad_flat.norm().item()

    result: dict[str, dict[str, float]] = {}
    present_codes = [
        code
        for code in range(len(CATEGORY_NAMES))
        if bool((category_weights[code].sum() > 0).item())
    ]

    for code, category in enumerate(CATEGORY_NAMES):
        budget = budget_by_category[code].item()

        if code not in present_codes:
            result[category] = {"grad_norm": 0.0, "cosine_sim": 0.0, "budget": budget}
            continue

        model.zero_grad(set_to_none=True)
        sub_loss = compute_aggregated_loss(
            old_logprobs,
            new_logprobs,
            category_weights[code, 0],
            category_weights[code, 1],
            cfg.loss_type,
        )
        sub_loss.backward(retain_graph=code != present_codes[-1])

        cat_grad_flat = flatten_current_grads(model)
        cat_norm = cat_grad_flat.norm().item()
        cos_sim = 0.0
        if total_norm > 0 and cat_norm > 0:
            cos_sim = (torch.dot(cat_grad_flat, total_grad_flat) / (cat_norm * total_norm)).item()

        result[category] = {"grad_norm": cat_norm, "cosine_sim": cos_sim, "budget": budget}

    total_budget = sum(info["budget"] for info in result.values())
    for category in CATEGORY_NAMES:
        result[category]["budget_fraction"] = (
            result[category]["budget"] / total_budget if total_budget > 0 else 0.0
        )

    model.zero_grad(set_to_none=True)
    for param, grad in zip(params, total_grads):
        param.grad = grad

    return result


def run_config_task(
    cfg: Config,
    *,
    n_steps: int,
    B: int,
    G: int,
    vocab_size: int,
    hidden: int,
    max_tok: int,
) -> tuple[str, dict[str, object]]:
    t0 = time.time()
    metrics, decomp = run_experiment(
        cfg,
        n_steps=n_steps,
        B=B,
        G=G,
        vocab_size=vocab_size,
        hidden=hidden,
        max_tok=max_tok,
    )
    key = f"{cfg.name}_s{cfg.seed}"
    return key, {"metrics": metrics, "grad_decomp": decomp, "time": time.time() - t0}


# ─── CLI / reporting ────────────────────────────────────────────────────────────

def build_shared_configs() -> list[Config]:
    shared_configs: list[Config] = []
    loss_aggs = ["token_sum", "per_traj_mean"]
    adv_schemes = ["maxrl", "mean_center"]
    seeds = [0, 1, 2]

    for loss_agg, adv_scheme in itertools.product(loss_aggs, adv_schemes):
        for seed in seeds:
            shared_configs.append(
                Config(
                    loss_agg=loss_agg,
                    adv_scheme=adv_scheme,
                    loss_type="ppo",
                    seed=seed,
                )
            )
    return shared_configs


def materialize_loss_type_results(
    shared_results: dict[tuple[str, str, int], dict[str, object]],
    loss_types: list[LossType],
) -> dict[str, dict[str, object]]:
    all_results: dict[str, dict[str, object]] = {}
    for (loss_agg, adv_scheme, seed), result in shared_results.items():
        shared_metrics = result["metrics"]
        assert isinstance(shared_metrics, list)
        for loss_type in loss_types:
            key = f"{loss_agg}_{adv_scheme}_{loss_type}_s{seed}"
            metrics = []
            for metric in shared_metrics:
                metrics.append(
                    {
                        "step": metric["step"],
                        "entropy": metric["entropy"],
                        "grad_norm": metric["grad_norm"],
                        "ref_logprob": metric["ref_logprob"],
                        "loss": metric["losses"][loss_type],
                    }
                )
            all_results[key] = {
                "metrics": metrics,
                "grad_decomp": result["grad_decomp"],
                "time": result["time"],
            }
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500, help="Training steps per config.")
    parser.add_argument("--max-tok", type=int, default=DEFAULT_MAX_TOK, help="Maximum effective token length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_configs()
    shared_configs = build_shared_configs()
    print(f"Running {len(shared_configs)} unique training trajectories for {len(configs)} reported configs...")
    shared_results: dict[tuple[str, str, int], dict[str, object]] = {}

    run_kwargs = {
        "n_steps": args.steps,
        "B": 16,
        "G": 8,
        "vocab_size": 100,
        "hidden": 256,
        "max_tok": args.max_tok,
    }

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    for i, cfg in enumerate(shared_configs, start=1):
        _, result = run_config_task(cfg, **run_kwargs)
        shared_results[(cfg.loss_agg, cfg.adv_scheme, cfg.seed)] = result
        print(f"  [{i}/{len(shared_configs)}] {cfg.loss_agg}_{cfg.adv_scheme}_all_losses_s{cfg.seed} done in {result['time']:.1f}s")

    loss_types: list[LossType] = ["ppo", "cispo", "reinforce"]
    all_results = materialize_loss_type_results(shared_results, loss_types)

    loss_aggs = ["token_sum", "per_traj_mean"]
    adv_schemes = ["maxrl", "mean_center"]
    seeds = [0, 1, 2]

    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    config_names = []
    for loss_agg, adv_scheme, loss_type in itertools.product(loss_aggs, adv_schemes, loss_types):
        config_names.append(f"{loss_agg}_{adv_scheme}_{loss_type}")

    header = f"{'Config':<40} {'Entropy':>16} {'Grad Norm':>16} {'Ref LogProb':>16} {'Stable?':>10}"
    print(header)
    print("-" * 120)

    summary_data = {}

    for config_name in config_names:
        seed_metrics = []
        for seed in seeds:
            key = f"{config_name}_s{seed}"
            metrics = all_results[key]["metrics"]
            seed_metrics.append(metrics[-1])

        entropies = [metric["entropy"] for metric in seed_metrics]
        grad_norms = [metric["grad_norm"] for metric in seed_metrics]
        ref_lps = [metric["ref_logprob"] for metric in seed_metrics]

        e_mean, e_std = np.mean(entropies), np.std(entropies)
        g_mean, g_std = np.mean(grad_norms), np.std(grad_norms)
        r_mean, r_std = np.mean(ref_lps), np.std(ref_lps)

        all_steps_entropy = []
        for seed in seeds:
            key = f"{config_name}_s{seed}"
            for metric in all_results[key]["metrics"]:
                all_steps_entropy.append(metric["entropy"])
        min_entropy = min(all_steps_entropy)
        max_grad_norm = max(
            metric["grad_norm"]
            for seed in seeds
            for metric in all_results[f"{config_name}_s{seed}"]["metrics"]
        )
        stable = "YES" if min_entropy > 0.5 and max_grad_norm < 1e6 else "UNSTABLE"

        summary_data[config_name] = {
            "entropy": (e_mean, e_std),
            "grad_norm": (g_mean, g_std),
            "ref_logprob": (r_mean, r_std),
            "stable": stable,
        }

        print(
            f"{config_name:<40} "
            f"{e_mean:>7.3f}+-{e_std:>5.3f}  "
            f"{g_mean:>7.3f}+-{g_std:>5.3f}  "
            f"{r_mean:>7.3f}+-{r_std:>5.3f}  "
            f"{stable:>10}"
        )

    print("\n" + "=" * 120)
    print("GRADIENT DECOMPOSITION (seed 0 only)")
    print("=" * 120)

    for step_label, step_key in [("Step 0", 0), ("Step 250", 250), ("Step 500", 499)]:
        print(f"\n--- {step_label} ---")
        print(f"{'Config':<40} {'Cat':<12} {'Grad Norm':>12} {'Cos Sim':>10} {'Budget Frac':>12}")
        print("-" * 90)
        for config_name in config_names:
            key = f"{config_name}_s0"
            decomp = all_results[key]["grad_decomp"]
            if step_key not in decomp:
                continue
            step_info = decomp[step_key]
            for category_name in CATEGORY_NAMES:
                info = step_info[category_name]
                label = config_name if category_name == "correct" else ""
                print(
                    f"{label:<40} {category_name:<12} "
                    f"{info['grad_norm']:>12.4f} {info['cosine_sim']:>10.4f} {info['budget_fraction']:>12.4f}"
                )

    print("\n" + "=" * 120)
    print("COMPARISON: maxrl vs mean_center")
    print("=" * 120)

    for loss_agg in loss_aggs:
        for loss_type in loss_types:
            maxrl_key = f"{loss_agg}_maxrl_{loss_type}"
            mean_center_key = f"{loss_agg}_mean_center_{loss_type}"
            maxrl_summary = summary_data[maxrl_key]
            mean_center_summary = summary_data[mean_center_key]
            delta_ent = maxrl_summary["entropy"][0] - mean_center_summary["entropy"][0]
            delta_gn = maxrl_summary["grad_norm"][0] - mean_center_summary["grad_norm"][0]
            delta_ref_lp = maxrl_summary["ref_logprob"][0] - mean_center_summary["ref_logprob"][0]
            print(
                f"{loss_agg:>15} {loss_type:>10}:  "
                f"Delta entropy={delta_ent:+.4f}  Delta grad_norm={delta_gn:+.4f}  Delta ref_lp={delta_ref_lp:+.4f}"
            )

    print("\n" + "=" * 120)
    print("COMPARISON: PPO vs CISPO vs REINFORCE")
    print("=" * 120)

    for loss_agg in loss_aggs:
        for adv_scheme in adv_schemes:
            print(f"\n  {loss_agg} / {adv_scheme}:")
            for loss_type in loss_types:
                key = f"{loss_agg}_{adv_scheme}_{loss_type}"
                summary = summary_data[key]
                print(
                    f"    {loss_type:>10}: "
                    f"ent={summary['entropy'][0]:.4f}  "
                    f"gn={summary['grad_norm'][0]:.4f}  "
                    f"ref_lp={summary['ref_logprob'][0]:.4f}  "
                    f"{summary['stable']}"
                )

    print("\n" + "=" * 120)
    print("DYNAMICS TRACES (step 0 vs step 490)")
    print("=" * 120)
    print(
        f"{'Config':<40} {'Step 0 Ent':>10} {'Step 490 Ent':>12} "
        f"{'Step 0 GN':>10} {'Step 490 GN':>12} {'Step 0 RL':>10} {'Step 490 RL':>12}"
    )
    print("-" * 120)
    for config_name in config_names:
        key = f"{config_name}_s0"
        metrics = all_results[key]["metrics"]
        first = metrics[0]
        last = metrics[-1]
        print(
            f"{config_name:<40} "
            f"{first['entropy']:>10.4f} {last['entropy']:>12.4f} "
            f"{first['grad_norm']:>10.4f} {last['grad_norm']:>12.4f} "
            f"{first['ref_logprob']:>10.4f} {last['ref_logprob']:>12.4f}"
        )

    out_path = Path(__file__).with_name("toy_ppo_matrix_results.json")
    json_results = {}
    for key, value in all_results.items():
        json_results[key] = {
            "metrics": value["metrics"],
            "grad_decomp": {str(step_key): step_value for step_key, step_value in value["grad_decomp"].items()},
            "time": value["time"],
        }
    with out_path.open("w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
