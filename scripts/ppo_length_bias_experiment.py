#!/usr/bin/env python3
"""
PPO training dynamics experiment: length bias under different advantage
and loss aggregation schemes.

Simulates a toy policy (2-layer MLP with position embeddings) generating
variable-length responses. Measures how advantage normalization and
per-token vs per-sequence loss aggregation interact with the length
distribution of correct, wrong, and truncated responses.

No external dependencies beyond torch and numpy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Literal
import sys

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters ──────────────────────────────────────────────────
VOCAB_SIZE = 100
HIDDEN_DIM = 128
MAX_LEN = 4096
NUM_STEPS = 200
BATCH_SIZE = 16  # number of groups per step
GROUP_SIZE = 8   # responses per group
LR = 1e-3

# Reward distribution
P_TRUNCATED = 0.35
P_WRONG = 0.50
P_CORRECT = 0.15
R_TRUNCATED = -0.2
R_WRONG = 0.0
R_CORRECT = 1.0

AdvScheme = Literal["mean_center", "maxrl"]
LossAgg = Literal["token_sum", "token_mean"]


# ── Model ────────────────────────────────────────────────────────────
class ToyPolicy(nn.Module):
    """2-layer MLP conditioned on position embedding. Outputs logits over vocab."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = HIDDEN_DIM,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (T,) → logits: (T, vocab_size)"""
        h = self.pos_embed(positions)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    def log_probs_and_entropy(self, positions: torch.Tensor,
                              tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions: (T,)
        tokens: (T,)  — the "generated" token ids
        Returns: log_probs (T,), entropy (T,)
        """
        logits = self.forward(positions)  # (T, V)
        log_p = F.log_softmax(logits, dim=-1)  # (T, V)
        token_log_probs = log_p.gather(1, tokens.unsqueeze(1)).squeeze(1)  # (T,)
        probs = log_p.exp()
        entropy = -(probs * log_p).sum(dim=-1)  # (T,)
        return token_log_probs, entropy


# ── Rollout generation ───────────────────────────────────────────────
@dataclass
class Episode:
    length: int
    reward: float
    category: str  # "truncated", "wrong", "correct"
    tokens: torch.Tensor     # (length,) — random token ids
    positions: torch.Tensor  # (length,) — 0..length-1


def sample_episodes(B: int, G: int) -> list[list[Episode]]:
    """Sample B groups of G episodes each."""
    groups = []
    for _ in range(B):
        group = []
        for _ in range(G):
            u = np.random.random()
            if u < P_TRUNCATED:
                length = MAX_LEN
                reward = R_TRUNCATED
                cat = "truncated"
            elif u < P_TRUNCATED + P_WRONG:
                length = int(np.clip(np.random.normal(2500, 800), 200, MAX_LEN - 1))
                reward = R_WRONG
                cat = "wrong"
            else:
                length = int(np.clip(np.random.normal(2500, 800), 200, MAX_LEN - 1))
                reward = R_CORRECT
                cat = "correct"

            tokens = torch.randint(0, VOCAB_SIZE, (length,))
            positions = torch.arange(length)
            group.append(Episode(length=length, reward=reward, category=cat,
                                 tokens=tokens, positions=positions))
        groups.append(group)
    return groups


# ── Advantage computation ────────────────────────────────────────────
def compute_advantages(groups: list[list[Episode]], scheme: AdvScheme) -> list[list[float]]:
    """Returns per-episode advantages, organized as groups[b][g]."""
    all_advs = []
    for group in groups:
        rewards = np.array([ep.reward for ep in group])
        mean_r = rewards.mean()
        std_r = rewards.std()
        min_r = rewards.min()
        max_r = rewards.max()
        rng = max_r - min_r

        if scheme == "mean_center":
            if std_r < 1e-8:
                advs = [0.0] * len(group)
            else:
                advs = [(r - mean_r) / std_r for r in rewards]
        elif scheme == "maxrl":
            if rng < 1e-8:
                advs = [0.0] * len(group)
            else:
                p_eff = (mean_r - min_r) / rng
                p_eff = max(p_eff, 1e-8)  # avoid division by zero
                advs = [(r - mean_r) / p_eff for r in rewards]
            # Note: maxrl doesn't divide by std, it divides by p_eff
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        all_advs.append(advs)
    return all_advs


# ── Loss computation ─────────────────────────────────────────────────
def compute_loss(model: ToyPolicy, groups: list[list[Episode]],
                 advantages: list[list[float]], agg: LossAgg,
                 return_decomposition: bool = False
                 ) -> tuple[torch.Tensor, float, dict | None]:
    """
    Compute policy gradient loss.
    Returns: (loss, mean_entropy, decomposition_dict_or_None)
    """
    total_loss = torch.tensor(0.0)
    total_entropy = 0.0
    total_tokens = 0

    # For gradient decomposition
    cat_losses: dict[str, torch.Tensor] = {}
    if return_decomposition:
        for cat in ("truncated", "wrong", "correct"):
            cat_losses[cat] = torch.tensor(0.0)

    for b, group in enumerate(groups):
        for g, ep in enumerate(group):
            adv = advantages[b][g]
            if abs(adv) < 1e-10:
                continue

            log_probs, entropy = model.log_probs_and_entropy(ep.positions, ep.tokens)

            if agg == "token_sum":
                # Sum of (advantage * log_prob) over tokens
                ep_loss = -(adv * log_probs.sum())
            elif agg == "token_mean":
                # Mean of (advantage * log_prob) over tokens, i.e. divide by seq len
                ep_loss = -(adv * log_probs.mean())
            else:
                raise ValueError(f"Unknown agg: {agg}")

            total_loss = total_loss + ep_loss
            total_entropy += entropy.mean().item() * ep.length
            total_tokens += ep.length

            if return_decomposition:
                cat_losses[ep.category] = cat_losses[ep.category] + ep_loss

    # Normalize by number of episodes (not tokens) for the outer sum
    n_episodes = sum(len(g) for g in groups)
    total_loss = total_loss / n_episodes
    mean_entropy = total_entropy / max(total_tokens, 1)

    decomp = None
    if return_decomposition:
        decomp = {k: (v / n_episodes).item() for k, v in cat_losses.items()}

    return total_loss, mean_entropy, decomp


# ── Training loop ────────────────────────────────────────────────────
@dataclass
class RunResult:
    adv_scheme: str
    loss_agg: str
    entropy_trajectory: list[float]
    loss_trajectory: list[float]
    grad_norm_trajectory: list[float]
    reward_trajectory: list[float]
    decomposition: dict[str, float] | None  # from final step


def run_experiment(adv_scheme: AdvScheme, loss_agg: LossAgg) -> RunResult:
    """Run a single training experiment."""
    # Fresh model for each run
    torch.manual_seed(SEED)
    model = ToyPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    entropy_traj = []
    loss_traj = []
    grad_norm_traj = []
    reward_traj = []
    decomp = None

    for step in range(NUM_STEPS):
        # Use a different seed per step but same across configs for fair comparison
        np.random.seed(SEED + step)

        groups = sample_episodes(BATCH_SIZE, GROUP_SIZE)
        advantages = compute_advantages(groups, adv_scheme)

        # Mean reward for tracking
        all_rewards = [ep.reward for g in groups for ep in g]
        reward_traj.append(float(np.mean(all_rewards)))

        # Compute loss (with decomposition on last step)
        is_last = (step == NUM_STEPS - 1)
        loss, mean_entropy, step_decomp = compute_loss(
            model, groups, advantages, loss_agg,
            return_decomposition=is_last
        )
        if is_last:
            decomp = step_decomp

        optimizer.zero_grad()
        loss.backward()

        # Gradient L2 norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        entropy_traj.append(mean_entropy)
        loss_traj.append(loss.item())
        grad_norm_traj.append(grad_norm)

        if step % 50 == 0 or step == NUM_STEPS - 1:
            print(f"  step {step:3d} | loss={loss.item():+10.2f} | "
                  f"entropy={mean_entropy:.3f} | grad_norm={grad_norm:.2f}")

    return RunResult(
        adv_scheme=adv_scheme,
        loss_agg=loss_agg,
        entropy_trajectory=entropy_traj,
        loss_trajectory=loss_traj,
        grad_norm_trajectory=grad_norm_traj,
        reward_trajectory=reward_traj,
        decomposition=decomp,
    )


# ── Gradient decomposition experiment ────────────────────────────────
def gradient_decomposition_experiment():
    """
    For a single batch, compute gradients separately from truncated, wrong,
    and correct episodes. Report the magnitude and cosine similarity.
    """
    print("\n" + "=" * 80)
    print("GRADIENT DECOMPOSITION EXPERIMENT")
    print("=" * 80)

    np.random.seed(SEED + 999)
    groups = sample_episodes(BATCH_SIZE, GROUP_SIZE)

    results = {}

    for adv_scheme in ["mean_center", "maxrl"]:
        for loss_agg in ["token_sum", "token_mean"]:
            config_key = f"{adv_scheme}/{loss_agg}"
            advantages = compute_advantages(groups, adv_scheme)

            torch.manual_seed(SEED)
            model = ToyPolicy()

            # Compute gradient from each category separately
            cat_grads: dict[str, torch.Tensor] = {}
            cat_token_counts: dict[str, int] = {"truncated": 0, "wrong": 0, "correct": 0}
            cat_episode_counts: dict[str, int] = {"truncated": 0, "wrong": 0, "correct": 0}
            cat_adv_sums: dict[str, float] = {"truncated": 0.0, "wrong": 0.0, "correct": 0.0}

            for cat in ("truncated", "wrong", "correct"):
                model.zero_grad()
                total_loss = torch.tensor(0.0)
                n_eps = 0

                for b, group in enumerate(groups):
                    for g, ep in enumerate(group):
                        if ep.category != cat:
                            continue
                        adv = advantages[b][g]
                        cat_adv_sums[cat] += adv
                        cat_episode_counts[cat] += 1
                        cat_token_counts[cat] += ep.length
                        if abs(adv) < 1e-10:
                            continue

                        log_probs, _ = model.log_probs_and_entropy(ep.positions, ep.tokens)
                        if loss_agg == "token_sum":
                            ep_loss = -(adv * log_probs.sum())
                        else:
                            ep_loss = -(adv * log_probs.mean())
                        total_loss = total_loss + ep_loss
                        n_eps += 1

                n_total = sum(len(g) for g in groups)
                total_loss = total_loss / n_total
                if n_eps > 0:
                    total_loss.backward()

                # Flatten all gradients into a single vector
                grad_vec = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_vec.append(p.grad.data.clone().flatten())
                    else:
                        grad_vec.append(torch.zeros(p.numel()))
                cat_grads[cat] = torch.cat(grad_vec)

            # Compute norms and cosine similarities
            norms = {cat: g.norm().item() for cat, g in cat_grads.items()}
            total_grad = sum(cat_grads.values())
            total_norm = total_grad.norm().item()

            # Cosine similarity between each category and total
            cosines = {}
            for cat in ("truncated", "wrong", "correct"):
                if norms[cat] > 1e-10 and total_norm > 1e-10:
                    cosines[cat] = (torch.dot(cat_grads[cat], total_grad) /
                                    (norms[cat] * total_norm)).item()
                else:
                    cosines[cat] = 0.0

            # Fraction of total gradient magnitude
            fractions = {cat: norms[cat] / max(total_norm, 1e-10) for cat in norms}

            results[config_key] = {
                "norms": norms,
                "total_norm": total_norm,
                "cosines": cosines,
                "fractions": fractions,
                "token_counts": cat_token_counts,
                "episode_counts": cat_episode_counts,
                "adv_sums": cat_adv_sums,
            }

    # Print results
    print(f"\n{'Config':<30} | {'Category':<12} | {'Grad Norm':>10} | "
          f"{'Frac of Total':>14} | {'Cos w/ Total':>12} | "
          f"{'#Episodes':>9} | {'#Tokens':>9} | {'Sum(adv)':>10}")
    print("-" * 140)

    for config_key, data in results.items():
        for cat in ("truncated", "wrong", "correct"):
            print(f"{config_key:<30} | {cat:<12} | "
                  f"{data['norms'][cat]:10.4f} | "
                  f"{data['fractions'][cat]:14.4f} | "
                  f"{data['cosines'][cat]:12.4f} | "
                  f"{data['episode_counts'][cat]:9d} | "
                  f"{data['token_counts'][cat]:9d} | "
                  f"{data['adv_sums'][cat]:+10.3f}")
        print(f"{'  TOTAL':<30} | {'---':<12} | "
              f"{data['total_norm']:10.4f} | {'1.0000':>14} | {'1.0000':>12} |"
              f"           |           |")
        print("-" * 140)


# ── Advantage statistics ─────────────────────────────────────────────
def advantage_statistics():
    """Print summary statistics of advantages under each scheme."""
    print("\n" + "=" * 80)
    print("ADVANTAGE STATISTICS (single batch)")
    print("=" * 80)

    np.random.seed(SEED + 999)
    groups = sample_episodes(BATCH_SIZE, GROUP_SIZE)

    for scheme in ["mean_center", "maxrl"]:
        advantages = compute_advantages(groups, scheme)

        # Collect per-category advantages
        cat_advs: dict[str, list[float]] = {"truncated": [], "wrong": [], "correct": []}
        for b, group in enumerate(groups):
            for g, ep in enumerate(group):
                cat_advs[ep.category].append(advantages[b][g])

        print(f"\n  Scheme: {scheme}")
        print(f"  {'Category':<12} | {'Count':>6} | {'Mean Adv':>10} | {'Std Adv':>10} | "
              f"{'Min':>10} | {'Max':>10} | {'|Sum|':>10}")
        print("  " + "-" * 80)
        for cat in ("truncated", "wrong", "correct"):
            advs = cat_advs[cat]
            if len(advs) == 0:
                continue
            a = np.array(advs)
            print(f"  {cat:<12} | {len(advs):6d} | {a.mean():+10.4f} | {a.std():10.4f} | "
                  f"{a.min():+10.4f} | {a.max():+10.4f} | {abs(a.sum()):10.4f}")


# ── Expected token-weighted advantage analysis ──────────────────────
def token_weighted_analysis():
    """
    Compute expected (advantage * tokens) per category to show how token_sum
    vs token_mean changes the effective weight of each category.
    """
    print("\n" + "=" * 80)
    print("TOKEN-WEIGHTED ADVANTAGE ANALYSIS (single batch)")
    print("=" * 80)

    np.random.seed(SEED + 999)
    groups = sample_episodes(BATCH_SIZE, GROUP_SIZE)

    for scheme in ["mean_center", "maxrl"]:
        advantages = compute_advantages(groups, scheme)

        cat_data: dict[str, list[tuple[float, int]]] = {
            "truncated": [], "wrong": [], "correct": []
        }
        for b, group in enumerate(groups):
            for g, ep in enumerate(group):
                cat_data[ep.category].append((advantages[b][g], ep.length))

        print(f"\n  Scheme: {scheme}")
        print(f"  {'Category':<12} | {'Sum(adv*len)':>14} | {'Sum(adv)':>12} | "
              f"{'Mean len':>10} | {'Effective weight ratio (sum/mean)':>34}")
        print("  " + "-" * 100)
        for cat in ("truncated", "wrong", "correct"):
            items = cat_data[cat]
            if not items:
                continue
            sum_adv_len = sum(a * l for a, l in items)
            sum_adv = sum(a for a, l in items)
            mean_len = np.mean([l for _, l in items])
            # Under token_sum, gradient ∝ adv * len; under token_mean, gradient ∝ adv
            ratio = sum_adv_len / sum_adv if abs(sum_adv) > 1e-10 else float('nan')
            print(f"  {cat:<12} | {sum_adv_len:+14.1f} | {sum_adv:+12.4f} | "
                  f"{mean_len:10.0f} | {ratio:34.0f}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    configs = [
        ("mean_center", "token_sum"),
        ("mean_center", "token_mean"),
        ("maxrl", "token_sum"),
        ("maxrl", "token_mean"),
    ]

    results: list[RunResult] = []

    for adv_scheme, loss_agg in configs:
        print(f"\n{'=' * 80}")
        print(f"RUNNING: adv={adv_scheme}, agg={loss_agg}")
        print(f"{'=' * 80}")
        result = run_experiment(adv_scheme, loss_agg)
        results.append(result)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Config':<30} | {'Entropy Start':>14} | {'Entropy End':>12} | "
          f"{'Entropy Δ':>10} | {'Mean GradNorm':>14} | "
          f"{'Med GradNorm':>13} | {'Dynamics':>12}")
    print("-" * 120)

    for r in results:
        config = f"{r.adv_scheme}/{r.loss_agg}"
        e_start = np.mean(r.entropy_trajectory[:5])
        e_end = np.mean(r.entropy_trajectory[-5:])
        e_delta = e_end - e_start
        mean_gn = np.mean(r.grad_norm_trajectory)
        med_gn = np.median(r.grad_norm_trajectory)

        # Classify dynamics
        loss_arr = np.array(r.loss_trajectory)
        first_quarter = np.mean(loss_arr[:50])
        last_quarter = np.mean(loss_arr[-50:])
        loss_std = np.std(loss_arr[-50:])

        if abs(last_quarter) > 10 * abs(first_quarter) and abs(last_quarter) > 100:
            dynamics = "DIVERGING"
        elif loss_std > abs(np.mean(loss_arr)) * 0.5 and loss_std > 10:
            dynamics = "OSCILLATING"
        else:
            dynamics = "CONVERGING"

        print(f"{config:<30} | {e_start:14.4f} | {e_end:12.4f} | "
              f"{e_delta:+10.4f} | {mean_gn:14.2f} | "
              f"{med_gn:13.2f} | {dynamics:>12}")

    # ── Loss decomposition from final step ───────────────────────────
    print(f"\n{'Config':<30} | {'Truncated Loss':>15} | {'Wrong Loss':>12} | "
          f"{'Correct Loss':>13} | {'Dominant':>12}")
    print("-" * 100)

    for r in results:
        config = f"{r.adv_scheme}/{r.loss_agg}"
        d = r.decomposition
        if d is None:
            print(f"{config:<30} | {'N/A':>15} | {'N/A':>12} | {'N/A':>13} | {'N/A':>12}")
            continue
        vals = {k: abs(v) for k, v in d.items()}
        dominant = max(vals, key=vals.get)
        print(f"{config:<30} | {d['truncated']:+15.4f} | {d['wrong']:+12.4f} | "
              f"{d['correct']:+13.4f} | {dominant:>12}")

    # ── Entropy trajectory details ───────────────────────────────────
    print(f"\n{'=' * 80}")
    print("ENTROPY TRAJECTORIES (every 20 steps)")
    print("=" * 80)
    steps_to_show = list(range(0, NUM_STEPS, 20)) + [NUM_STEPS - 1]
    header = f"{'Step':>5}"
    for r in results:
        config = f"{r.adv_scheme}/{r.loss_agg}"
        header += f" | {config:>20}"
    print(header)
    print("-" * (6 + 23 * len(results)))
    for s in steps_to_show:
        row = f"{s:5d}"
        for r in results:
            row += f" | {r.entropy_trajectory[s]:20.4f}"
        print(row)

    # ── Gradient norm trajectory ─────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("GRADIENT NORM TRAJECTORIES (every 20 steps)")
    print("=" * 80)
    print(header)
    print("-" * (6 + 23 * len(results)))
    for s in steps_to_show:
        row = f"{s:5d}"
        for r in results:
            row += f" | {r.grad_norm_trajectory[s]:20.4f}"
        print(row)

    # ── Additional analyses ──────────────────────────────────────────
    advantage_statistics()
    token_weighted_analysis()
    gradient_decomposition_experiment()

    # ── Final analysis ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RATIO ANALYSIS: token_sum vs token_mean gradient norms")
    print("=" * 80)
    for adv in ["mean_center", "maxrl"]:
        sum_result = [r for r in results if r.adv_scheme == adv and r.loss_agg == "token_sum"][0]
        mean_result = [r for r in results if r.adv_scheme == adv and r.loss_agg == "token_mean"][0]
        ratio = np.mean(sum_result.grad_norm_trajectory) / max(np.mean(mean_result.grad_norm_trajectory), 1e-10)
        print(f"  {adv}: mean(grad_norm_sum) / mean(grad_norm_mean) = {ratio:.1f}x")
        print(f"    token_sum  grad_norm: mean={np.mean(sum_result.grad_norm_trajectory):.2f}, "
              f"std={np.std(sum_result.grad_norm_trajectory):.2f}")
        print(f"    token_mean grad_norm: mean={np.mean(mean_result.grad_norm_trajectory):.2f}, "
              f"std={np.std(mean_result.grad_norm_trajectory):.2f}")


if __name__ == "__main__":
    main()
