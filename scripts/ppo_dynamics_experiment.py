#!/usr/bin/env python3
"""
PPO Training Dynamics with Variable-Length Responses — Definitive Experiment.

Uses the REAL _normalize_subgroup from tinker_cookbook.rl.data_processing.
Reward distribution derived from actual metrics.jsonl (gpqa-g8-s42).

12 configs = 2 aggregations × 2 advantages × 3 losses, each with 3 seeds.

SCALING NOTE: Real training uses a ~35B-param model with LR=1e-5, grad_clip=0.3.
Our proxy model has ~55K params. We preserve the ratio:
  - pre_clip_norm / clip_norm ≈ same (both heavily clip)
  - effective_lr * clip_norm ≈ scaled to produce visible dynamics
The gradient DIRECTION and relative category contributions are model-size-invariant.
"""

import sys
import os
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import the REAL _normalize_subgroup from the codebase
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinker_cookbook.rl.data_processing import _normalize_subgroup

# ---------------------------------------------------------------------------
# Reward distribution from actual GPQA data (gpqa-g8-s42/metrics.jsonl)
# ---------------------------------------------------------------------------
# Verified from 20 steps × 512 episodes each:
#   Mean correct:          0.378  → reward = 1.0
#   Mean wrong_formatted:  0.313  → reward = 0.0  (format_eos - correct)
#   Mean truncated:        0.309  → reward = -0.2  (1 - format_eos)
# (format_coef=0.1, eos_coef=0.1 per config.json)
P_CORRECT = 0.378
P_WRONG = 0.313
P_TRUNCATED = 0.309

R_CORRECT = 1.0
R_WRONG = 0.0
R_TRUNCATED = -0.2

# Length distribution from metrics.jsonl:
#   Mean ac_tokens_per_turn ≈ 4016 across all categories.
#   max_tokens = 8192 in config.json.
#   Truncated = exactly max_tokens. Non-truncated: solving
#     4016 = 0.378*L_c + 0.313*L_w + 0.309*8192  →  L_nontrun ≈ 2150
# Simulation lengths (scaled by 1/32 for tractability, ratio preserved ~4x):
SIM_LEN_CORRECT = 62
SIM_LEN_WRONG = 71
SIM_MAX_LEN = 256

# ---------------------------------------------------------------------------
# LR / grad_clip scaling
# ---------------------------------------------------------------------------
# Real: ~35B params, LR=1e-5, grad_clip=0.3, pre-clip norm ~20K
# Sim:  ~55K params, same loss structure
# We want the policy to actually evolve over 500 steps so differences are visible.
# The grad clip ratio (pre/clip ≈ 67000x in real training) means Adam's adaptive
# LR is the true bottleneck. We use LR=3e-4 with grad_clip=1.0 — enough to
# see dynamics while still clipping (pre-clip norms are ~100-1000 in our model).
SIM_LR = 3e-4
SIM_GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Model: 2-layer MLP
# ---------------------------------------------------------------------------
class PolicyMLP(nn.Module):
    def __init__(self, vocab_size=100, hidden=256, context_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_dim = context_dim
        self.fc1 = nn.Linear(vocab_size + context_dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch, seq_len) int → (B, T, V) one-hot
        positions: (batch, seq_len) float
        Returns: (batch, seq_len, vocab_size) log-probs
        """
        one_hot = F.one_hot(tokens.long(), self.vocab_size).float()
        pos_feat = positions.unsqueeze(-1).expand(-1, -1, self.context_dim)
        freqs = torch.arange(self.context_dim, device=tokens.device).float() * 0.1
        pos_enc = torch.sin(pos_feat * freqs)
        x = torch.cat([one_hot, pos_enc], dim=-1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    tokens: torch.Tensor       # (seq_len,) int
    log_probs: torch.Tensor    # (seq_len,) float — under sampling policy
    reward: float
    category: str              # "correct", "wrong", "truncated"
    length: int


def generate_trajectory(category: str, vocab_size: int, rng: np.random.Generator) -> Trajectory:
    if category == "correct":
        mean_len, reward = SIM_LEN_CORRECT, R_CORRECT
    elif category == "wrong":
        mean_len, reward = SIM_LEN_WRONG, R_WRONG
    else:
        mean_len, reward = SIM_MAX_LEN, R_TRUNCATED

    if category == "truncated":
        length = SIM_MAX_LEN
    else:
        length = max(10, int(rng.normal(mean_len, mean_len * 0.3)))
        length = min(length, SIM_MAX_LEN - 1)

    tokens = torch.tensor(rng.integers(0, vocab_size, size=length), dtype=torch.long)
    log_probs = torch.zeros(length)
    return Trajectory(tokens=tokens, log_probs=log_probs, reward=reward,
                      category=category, length=length)


def generate_batch(B: int, G: int, vocab_size: int, rng: np.random.Generator) -> list[list[Trajectory]]:
    groups = []
    cats_arr = ["correct", "wrong", "truncated"]
    probs = [P_CORRECT, P_WRONG, P_TRUNCATED]
    for _ in range(B):
        cats = rng.choice(cats_arr, size=G, p=probs)
        groups.append([generate_trajectory(c, vocab_size, rng) for c in cats])
    return groups


# ---------------------------------------------------------------------------
# Log-probs under current policy
# ---------------------------------------------------------------------------
def compute_log_probs_for_traj(model: PolicyMLP, traj: Trajectory) -> torch.Tensor:
    tokens = traj.tokens.unsqueeze(0)
    positions = torch.arange(traj.length, dtype=torch.float32).unsqueeze(0) / SIM_MAX_LEN
    with torch.no_grad():
        lp_all = model(tokens, positions)
    return lp_all[0, torch.arange(traj.length), traj.tokens]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
AggregationType = Literal["token_sum", "per_traj_mean"]


def compute_loss(
    model: PolicyMLP,
    trajs: list[Trajectory],
    advantages: torch.Tensor,
    loss_type: str,
    aggregation: AggregationType,
    clip_low: float = 0.8,
    clip_high: float = 1.28,
) -> torch.Tensor:
    """
    token_sum:     loss_traj = sum_t L(t)      → longer trajs contribute more
    per_traj_mean: loss_traj = mean_t L(t)     → normalizes out length
    Total loss = sum over trajectories of loss_traj.
    """
    total_loss = torch.tensor(0.0)

    for i, traj in enumerate(trajs):
        adv = advantages[i].item()
        if abs(adv) < 1e-10:
            continue

        tokens = traj.tokens.unsqueeze(0)
        positions = torch.arange(traj.length, dtype=torch.float32).unsqueeze(0) / SIM_MAX_LEN
        lp_all = model(tokens, positions)
        current_lp = lp_all[0, torch.arange(traj.length), traj.tokens]
        old_lp = traj.log_probs.detach()
        log_ratio = current_lp - old_lp

        if loss_type == "REINFORCE":
            per_token_loss = -adv * current_lp
        elif loss_type == "PPO":
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, clip_low, clip_high)
            if adv > 0:
                surrogate = torch.min(ratio * adv, clipped * adv)
            else:
                surrogate = torch.max(ratio * adv, clipped * adv)
            per_token_loss = -surrogate
        elif loss_type == "CISPO":
            log_cl, log_ch = math.log(clip_low), math.log(clip_high)
            clipped_lr = torch.clamp(log_ratio, log_cl, log_ch)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.exp(clipped_lr)
            if adv > 0:
                surrogate = torch.min(ratio * adv, clipped_ratio * adv)
            else:
                surrogate = torch.max(ratio * adv, clipped_ratio * adv)
            per_token_loss = -surrogate
        else:
            raise ValueError(f"Unknown loss: {loss_type}")

        if aggregation == "token_sum":
            total_loss = total_loss + per_token_loss.sum()
        else:
            total_loss = total_loss + per_token_loss.mean()

    return total_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_entropy(model: PolicyMLP, ref_tokens: torch.Tensor, ref_positions: torch.Tensor) -> float:
    with torch.no_grad():
        lp = model(ref_tokens.unsqueeze(0), ref_positions.unsqueeze(0))
        probs = torch.exp(lp[0])
        return -(probs * lp[0]).sum(dim=-1).mean().item()


def compute_ref_logprob(model: PolicyMLP, ref_tokens: torch.Tensor, ref_positions: torch.Tensor) -> float:
    with torch.no_grad():
        lp = model(ref_tokens.unsqueeze(0), ref_positions.unsqueeze(0))
        return lp[0, torch.arange(len(ref_tokens)), ref_tokens].mean().item()


def grad_l2_norm(model: PolicyMLP) -> float:
    return math.sqrt(sum(p.grad.data.norm(2).item()**2
                         for p in model.parameters() if p.grad is not None))


def clip_grad_norm_(model: PolicyMLP, max_norm: float) -> tuple[float, float]:
    pre = grad_l2_norm(model)
    if pre > max_norm > 0:
        scale = max_norm / pre
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale)
    post = grad_l2_norm(model)
    return pre, post


# ---------------------------------------------------------------------------
# Gradient decomposition
# ---------------------------------------------------------------------------
def gradient_decomposition(
    model: PolicyMLP,
    groups: list[list[Trajectory]],
    advantage_scheme: str,
    aggregation: AggregationType,
    loss_type: str,
) -> dict:
    """Decompose gradient into contributions from correct/wrong/truncated."""
    all_trajs = []
    all_advs = []
    for group in groups:
        rewards = torch.tensor([t.reward for t in group], dtype=torch.float32)
        advs = _normalize_subgroup(rewards, advantage_scheme, alpha=1.0)
        all_trajs.extend(group)
        all_advs.append(advs)
    all_advs = torch.cat(all_advs)

    categories = {"correct": [], "wrong": [], "truncated": []}
    cat_advs = {"correct": [], "wrong": [], "truncated": []}
    for i, traj in enumerate(all_trajs):
        categories[traj.category].append(traj)
        cat_advs[traj.category].append(all_advs[i])

    cat_grads = {}
    cat_budgets = {}
    for cat in ["correct", "wrong", "truncated"]:
        if not categories[cat]:
            cat_grads[cat] = None
            cat_budgets[cat] = {"token_sum": 0.0, "per_traj_mean": 0.0}
            continue
        model.zero_grad()
        advs_t = torch.tensor([a.item() for a in cat_advs[cat]])
        loss = compute_loss(model, categories[cat], advs_t, loss_type, aggregation)
        loss.backward()
        cat_grads[cat] = torch.cat([p.grad.data.flatten()
                                     for p in model.parameters() if p.grad is not None]).clone()
        abs_advs = [abs(a.item()) for a in cat_advs[cat]]
        lens = [t.length for t in categories[cat]]
        cat_budgets[cat] = {
            "token_sum": sum(a * l for a, l in zip(abs_advs, lens)),
            "per_traj_mean": sum(abs_advs),
        }

    model.zero_grad()
    loss = compute_loss(model, all_trajs, all_advs, loss_type, aggregation)
    loss.backward()
    total_grad = torch.cat([p.grad.data.flatten()
                             for p in model.parameters() if p.grad is not None])

    result = {"total_grad_norm": total_grad.norm().item()}
    for cat in ["correct", "wrong", "truncated"]:
        if cat_grads[cat] is not None:
            result[f"{cat}_grad_norm"] = cat_grads[cat].norm().item()
            result[f"{cat}_cos_sim"] = F.cosine_similarity(
                cat_grads[cat].unsqueeze(0), total_grad.unsqueeze(0)).item()
        else:
            result[f"{cat}_grad_norm"] = 0.0
            result[f"{cat}_cos_sim"] = 0.0
        result[f"{cat}_budget"] = cat_budgets.get(cat, {"token_sum": 0, "per_traj_mean": 0})

    return result


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    aggregation: AggregationType
    advantage: str
    loss_type: str
    seed: int
    n_steps: int = 500
    B: int = 16
    G: int = 8
    lr: float = SIM_LR
    grad_clip: float = SIM_GRAD_CLIP
    vocab_size: int = 100


@dataclass
class StepMetrics:
    step: int
    entropy: float
    grad_norm_pre: float
    grad_norm_post: float
    ref_logprob: float


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    metrics: list[StepMetrics] = field(default_factory=list)
    grad_decomp: dict = field(default_factory=dict)
    stability: str = "unknown"


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = PolicyMLP(vocab_size=cfg.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ref_tokens = torch.tensor(rng.integers(0, cfg.vocab_size, size=50), dtype=torch.long)
    ref_positions = torch.arange(50, dtype=torch.float32) / SIM_MAX_LEN

    result = ExperimentResult(config=cfg)
    decomp_steps = {0, 100, 300, 499}

    for step in range(cfg.n_steps):
        groups = generate_batch(cfg.B, cfg.G, cfg.vocab_size, rng)

        # Compute old log-probs
        for group in groups:
            for traj in group:
                traj.log_probs = compute_log_probs_for_traj(model, traj)

        # Compute advantages using REAL _normalize_subgroup
        all_trajs = []
        all_advs = []
        for group in groups:
            rewards = torch.tensor([t.reward for t in group], dtype=torch.float32)
            advs = _normalize_subgroup(rewards, cfg.advantage, alpha=1.0)
            all_trajs.extend(group)
            all_advs.append(advs)
        all_advs_tensor = torch.cat(all_advs)

        # Gradient decomposition at key steps
        if step in decomp_steps:
            result.grad_decomp[step] = gradient_decomposition(
                model, groups, cfg.advantage, cfg.aggregation, cfg.loss_type
            )

        # Training step
        optimizer.zero_grad()
        loss = compute_loss(model, all_trajs, all_advs_tensor,
                           cfg.loss_type, cfg.aggregation)

        if torch.isnan(loss) or torch.isinf(loss):
            # Early termination on NaN
            result.stability = "DIVERGED"
            break

        loss.backward()
        pre_norm, post_norm = clip_grad_norm_(model, cfg.grad_clip)
        optimizer.step()

        if step % 10 == 0:
            ent = compute_entropy(model, ref_tokens, ref_positions)
            rlp = compute_ref_logprob(model, ref_tokens, ref_positions)
            result.metrics.append(StepMetrics(step, ent, pre_norm, post_norm, rlp))

            if math.isnan(ent) or math.isinf(ent):
                result.stability = "DIVERGED"
                break

    # Final
    ent = compute_entropy(model, ref_tokens, ref_positions)
    rlp = compute_ref_logprob(model, ref_tokens, ref_positions)
    result.metrics.append(StepMetrics(cfg.n_steps, ent, 0, 0, rlp))

    if result.stability == "unknown":
        entropies = [m.entropy for m in result.metrics]
        if any(math.isnan(e) or math.isinf(e) for e in entropies):
            result.stability = "DIVERGED"
        elif entropies[-1] < 0.5:
            result.stability = "COLLAPSED"
        elif max(entropies) - min(entropies) > 3.0:
            result.stability = "UNSTABLE"
        else:
            result.stability = "STABLE"

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_pm(vals):
    m, s = np.mean(vals), np.std(vals)
    return f"{m:+.4f}+/-{s:.4f}" if any(v < 0 for v in vals) else f"{m:.4f}+/-{s:.4f}"


def print_summary_table(results):
    print("\n" + "=" * 170)
    print(f"{'SUMMARY TABLE':^170}")
    print("=" * 170)
    hdr = (f"{'Config':<35} {'Start H':>14} {'End H':>14} {'dH':>14} "
           f"{'Grad Pre':>14} {'Grad Post':>14} {'RefLP':>14} {'Stability':>14}")
    print(hdr)
    print("-" * 170)

    for key in sorted(results):
        runs = results[key]
        sh = [r.metrics[0].entropy for r in runs]
        eh = [r.metrics[-1].entropy for r in runs]
        dh = [e - s for s, e in zip(sh, eh)]
        gp = [np.mean([m.grad_norm_pre for m in r.metrics[1:-1] if m.grad_norm_pre > 0])
              if len(r.metrics) > 2 else 0 for r in runs]
        gc = [np.mean([m.grad_norm_post for m in r.metrics[1:-1] if m.grad_norm_post > 0])
              if len(r.metrics) > 2 else 0 for r in runs]
        rl = [r.metrics[-1].ref_logprob for r in runs]
        stabs = set(r.stability for r in runs)
        print(f"{key:<35} {fmt_pm(sh):>14} {fmt_pm(eh):>14} {fmt_pm(dh):>14} "
              f"{fmt_pm(gp):>14} {fmt_pm(gc):>14} {fmt_pm(rl):>14} {'/'.join(stabs):>14}")


def print_grad_decomp(results, step):
    print(f"\n{'=' * 150}")
    print(f"{'GRADIENT DECOMPOSITION at step ' + str(step):^150}")
    print(f"{'=' * 150}")
    hdr = (f"{'Config':<35} {'||g||':>10} {'||g_c||':>10} {'||g_w||':>10} "
           f"{'||g_tr||':>10} {'cos(c,g)':>9} {'cos(w,g)':>9} {'cos(tr,g)':>9}")
    print(hdr)
    print("-" * 150)

    for key in sorted(results):
        runs = results[key]
        decomps = [r.grad_decomp.get(step) for r in runs if step in r.grad_decomp]
        if not decomps:
            continue
        def a(f):
            return np.mean([d[f] for d in decomps if d])
        print(f"{key:<35} {a('total_grad_norm'):>10.2f} {a('correct_grad_norm'):>10.2f} "
              f"{a('wrong_grad_norm'):>10.2f} {a('truncated_grad_norm'):>10.2f} "
              f"{a('correct_cos_sim'):>9.3f} {a('wrong_cos_sim'):>9.3f} "
              f"{a('truncated_cos_sim'):>9.3f}")


def print_budget_table(results, step):
    print(f"\n{'=' * 100}")
    print(f"{'GRADIENT BUDGET (% by category) at step ' + str(step):^100}")
    print(f"{'=' * 100}")
    # Show BOTH budget modes for every config
    hdr = f"{'Config':<35} {'Mode':<15} {'Correct%':>10} {'Wrong%':>10} {'Truncated%':>10}"
    print(hdr)
    print("-" * 100)

    for key in sorted(results):
        runs = results[key]
        decomps = [r.grad_decomp.get(step) for r in runs if step in r.grad_decomp]
        if not decomps:
            continue
        for mode in ["token_sum", "per_traj_mean"]:
            pcts = defaultdict(list)
            for d in decomps:
                if not d:
                    continue
                tot = sum(d[f"{c}_budget"][mode] for c in ["correct", "wrong", "truncated"])
                if tot > 0:
                    for c in ["correct", "wrong", "truncated"]:
                        pcts[c].append(d[f"{c}_budget"][mode] / tot * 100)
            if pcts:
                print(f"{key:<35} {mode:<15} "
                      f"{np.mean(pcts['correct']):>9.1f}% "
                      f"{np.mean(pcts['wrong']):>9.1f}% "
                      f"{np.mean(pcts['truncated']):>9.1f}%")


def print_maxrl_vs_mc(results):
    print(f"\n{'=' * 110}")
    print(f"{'MAXRL vs MEAN_CENTER':^110}")
    print(f"{'=' * 110}")

    pairs = defaultdict(dict)
    for key, runs in results.items():
        c = runs[0].config
        pairs[f"{c.aggregation}/{c.loss_type}"][c.advantage] = runs

    print(f"{'Agg/Loss':<28} {'Metric':<12} {'maxrl':>18} {'mean_center':>18} {'delta':>10} {'Sig?':>6}")
    print("-" * 110)

    any_sig = False
    for pk in sorted(pairs):
        if "maxrl" not in pairs[pk] or "mean_center" not in pairs[pk]:
            continue
        mr, mc = pairs[pk]["maxrl"], pairs[pk]["mean_center"]
        for name, fn in [("End H", lambda r: r.metrics[-1].entropy),
                         ("RefLP", lambda r: r.metrics[-1].ref_logprob),
                         ("dH", lambda r: r.metrics[-1].entropy - r.metrics[0].entropy)]:
            mv = [fn(r) for r in mr]
            cv = [fn(r) for r in mc]
            mm, sm = np.mean(mv), np.std(mv)
            mc_, sc = np.mean(cv), np.std(cv)
            d = abs(mm - mc_)
            ps = math.sqrt((sm**2 + sc**2) / 2) if (sm + sc) > 0 else 0
            sig = d > 2 * ps if ps > 0 else d > 0.01
            if sig:
                any_sig = True
            print(f"{pk:<28} {name:<12} {mm:>7.4f}+/-{sm:.4f} {mc_:>7.4f}+/-{sc:.4f} "
                  f"{d:>9.4f} {'YES' if sig else ' no':>5}")

    print(f"\nAny significant difference: {'YES' if any_sig else 'NO'}")


def print_stability_check(results):
    print(f"\n{'=' * 80}")
    print("STABILITY BY AGGREGATION")
    print(f"{'=' * 80}")
    for agg_label in ["per_traj_mean", "token_sum"]:
        print(f"\n  {agg_label.upper()}:")
        for key in sorted(results):
            runs = results[key]
            if runs[0].config.aggregation != agg_label:
                continue
            stabs = [r.stability for r in runs]
            ok = all(s == "STABLE" for s in stabs)
            print(f"    {key:<35} {'/'.join(stabs):>20}  {'OK' if ok else 'PROBLEM'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    AGGREGATIONS = ["token_sum", "per_traj_mean"]
    ADVANTAGES = ["maxrl", "mean_center"]
    LOSSES = ["PPO", "CISPO", "REINFORCE"]
    SEEDS = [42, 137, 2024]
    N_STEPS = 500

    total = len(AGGREGATIONS) * len(ADVANTAGES) * len(LOSSES) * len(SEEDS)
    print(f"Running {total} experiments "
          f"({len(AGGREGATIONS)} agg x {len(ADVANTAGES)} adv x {len(LOSSES)} loss x {len(SEEDS)} seeds)")
    print(f"Steps={N_STEPS}, B=16, G=8, LR={SIM_LR}, grad_clip={SIM_GRAD_CLIP}")
    print(f"Sim lengths: correct~{SIM_LEN_CORRECT}, wrong~{SIM_LEN_WRONG}, truncated={SIM_MAX_LEN}")
    print(f"Length ratio truncated/correct = {SIM_MAX_LEN/SIM_LEN_CORRECT:.1f}x")
    print(f"Reward: correct={R_CORRECT}, wrong={R_WRONG}, truncated={R_TRUNCATED}")
    print(f"Probs: correct={P_CORRECT}, wrong={P_WRONG}, truncated={P_TRUNCATED}")
    print()

    all_results: dict[str, list[ExperimentResult]] = defaultdict(list)
    n = 0

    for agg in AGGREGATIONS:
        for adv in ADVANTAGES:
            for loss in LOSSES:
                for seed in SEEDS:
                    n += 1
                    key = f"{agg}/{adv}/{loss}"
                    cfg = ExperimentConfig(
                        aggregation=agg, advantage=adv, loss_type=loss,
                        seed=seed, n_steps=N_STEPS
                    )
                    dt = time.time() - t0
                    print(f"  [{n:>2}/{total}] {key:<35} seed={seed}  ({dt:.0f}s)",
                          end="", flush=True)
                    r = run_experiment(cfg)
                    print(f"  {r.stability:<10} H:{r.metrics[0].entropy:.3f}->{r.metrics[-1].entropy:.3f}")
                    all_results[key].append(r)

    print_summary_table(all_results)
    for s in [0, 499]:
        print_grad_decomp(all_results, s)
        print_budget_table(all_results, s)
    print_maxrl_vs_mc(all_results)
    print_stability_check(all_results)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
