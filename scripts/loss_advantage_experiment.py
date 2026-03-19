"""
Tensor-level experiment: loss functions x advantage schemes x aggregation.

Loss functions: PPO (0.8/1.28), CISPO (0.8/1.28), REINFORCE (importance_sampling)
Advantage schemes: maxrl, mean_center
Aggregation: token_sum, per_traj_mean

CRITICAL DESIGN: For the loss functions to differ, we need OFF-POLICY ratios.
We simulate this by keeping a stale "sampler" model that is updated less frequently
than the learner (every K steps), matching real training where we sample a batch,
then take multiple gradient steps on it.

4 experiments:
  1. Gradient structure decomposition (single batch, off-policy)
  2. Training dynamics (300 steps, 12 configs)
  3. CISPO-specific ratio analysis
  4. Interaction effects table
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import time
from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)

VOCAB = 100
HIDDEN = 128
CLIP_LOW, CLIP_HIGH = 0.8, 1.28


# ─────────────────────────────────────────────
# Toy model
# ─────────────────────────────────────────────

class ToyPolicy(nn.Module):
    def __init__(self, vocab=VOCAB, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def log_probs(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens)
        log_p = F.log_softmax(logits, dim=-1)
        return log_p.gather(1, actions.unsqueeze(1)).squeeze(1)

    def entropy_mean(self, tokens: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(tokens)
            probs = F.softmax(logits, dim=-1)
            return -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()


# ─────────────────────────────────────────────
# Trajectory
# ─────────────────────────────────────────────

@dataclass
class Trajectory:
    tokens: torch.Tensor
    actions: torch.Tensor
    sampler_lp: torch.Tensor  # log-probs from the SAMPLER (stale) model
    reward: float
    length: int
    category: str


def sample_trajectory_with_model(rng: np.random.RandomState, sampler_model: nn.Module,
                                  max_len: int = 200) -> Trajectory:
    """Sample trajectory and record sampler log-probs from the given model."""
    u = rng.random()
    if u < 0.35:
        reward, length, cat = -0.2, max_len, "truncated"
    elif u < 0.85:
        length = int(np.clip(rng.normal(max_len * 0.6, max_len * 0.2), max_len // 8, max_len))
        reward, cat = 0.0, "wrong"
    else:
        length = int(np.clip(rng.normal(max_len * 0.6, max_len * 0.2), max_len // 8, max_len))
        reward, cat = 1.0, "correct"

    tokens = torch.randint(0, VOCAB, (length,))
    actions = torch.randint(0, VOCAB, (length,))
    with torch.no_grad():
        sampler_lp = sampler_model.log_probs(tokens, actions).detach()
    return Trajectory(tokens=tokens, actions=actions, sampler_lp=sampler_lp,
                      reward=reward, length=length, category=cat)


def generate_batch_with_model(B: int, G: int, rng: np.random.RandomState,
                               sampler_model: nn.Module,
                               max_len: int = 200) -> List[List[Trajectory]]:
    return [[sample_trajectory_with_model(rng, sampler_model, max_len)
             for _ in range(G)] for _ in range(B)]


# ─────────────────────────────────────────────
# Advantage computation
# ─────────────────────────────────────────────

def compute_advantages_maxrl(rewards: List[float]) -> List[float]:
    rt = torch.tensor(rewards, dtype=torch.float32)
    mean = rt.mean()
    centered = rt - mean
    r_min, r_max = rt.min(), rt.max()
    r_range = r_max - r_min
    if r_range < 1e-8:
        return [0.0] * len(rewards)
    p_eff = (mean - r_min) / r_range
    if p_eff <= 1e-8:
        return [0.0] * len(rewards)
    return (centered / p_eff).tolist()


def compute_advantages_mean_center(rewards: List[float]) -> List[float]:
    rt = torch.tensor(rewards, dtype=torch.float32)
    mean = rt.mean()
    std = rt.std()
    if std < 1e-8:
        return [0.0] * len(rewards)
    return ((rt - mean) / std).tolist()


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def reinforce_loss(learner_lp, sampler_lp, advantages):
    ratio = torch.exp(learner_lp - sampler_lp)
    return -(ratio * advantages).sum()


def ppo_loss(learner_lp, sampler_lp, advantages):
    ratio = torch.exp(learner_lp - sampler_lp)
    clipped = torch.clamp(ratio, CLIP_LOW, CLIP_HIGH)
    return -torch.min(ratio * advantages, clipped * advantages).sum()


def cispo_loss(learner_lp, sampler_lp, advantages):
    ratio = torch.exp(learner_lp - sampler_lp)
    clipped = torch.clamp(ratio, CLIP_LOW, CLIP_HIGH).detach()
    return -(clipped * learner_lp * advantages).sum()


LOSS_FNS = {"REINFORCE": reinforce_loss, "PPO": ppo_loss, "CISPO": cispo_loss}
ADV_FNS = {"maxrl": compute_advantages_maxrl, "mean_center": compute_advantages_mean_center}


# ─────────────────────────────────────────────
# EXPERIMENT 1: Gradient structure decomposition
# Uses a model that has been trained for 20 steps (off-policy ratios != 1)
# ─────────────────────────────────────────────

def experiment_1():
    print("=" * 80)
    print("EXPERIMENT 1: Gradient structure — PPO vs CISPO vs REINFORCE x advantage scheme")
    print("  Single batch, off-policy (sampler = init model, learner = 20-step-trained model)")
    print("=" * 80)

    MAX_LEN = 500  # longer for exp1 gradient analysis

    # Create sampler model (init) and train a learner for 20 steps
    torch.manual_seed(42)
    sampler_model = ToyPolicy()
    for p in sampler_model.parameters():
        p.requires_grad_(False)

    learner_model = ToyPolicy()
    learner_model.load_state_dict(sampler_model.state_dict())
    opt = torch.optim.Adam(learner_model.parameters(), lr=1e-3)

    rng_pretrain = np.random.RandomState(0)
    for _ in range(20):
        groups = generate_batch_with_model(16, 8, rng_pretrain, sampler_model, max_len=MAX_LEN)
        opt.zero_grad()
        total_loss = torch.tensor(0.0)
        for group in groups:
            rewards = [t.reward for t in group]
            advs = compute_advantages_mean_center(rewards)
            for traj, adv_val in zip(group, advs):
                lp = learner_model.log_probs(traj.tokens, traj.actions)
                adv_t = torch.full((traj.length,), adv_val)
                total_loss = total_loss + reinforce_loss(lp, traj.sampler_lp, adv_t)
        total_loss.backward()
        opt.step()

    # Check how off-policy we are
    rng_analysis = np.random.RandomState(42)
    groups = generate_batch_with_model(16, 8, rng_analysis, sampler_model, max_len=MAX_LEN)
    all_ratios = []
    for group in groups:
        for traj in group:
            with torch.no_grad():
                lp = learner_model.log_probs(traj.tokens, traj.actions)
                ratio = torch.exp(lp - traj.sampler_lp)
                all_ratios.append(ratio)
    r_all = torch.cat(all_ratios)
    pcts = np.percentile(r_all.numpy(), [5, 25, 50, 75, 95])
    frac_clipped = ((r_all < CLIP_LOW) | (r_all > CLIP_HIGH)).float().mean().item()
    print(f"\n  Off-policy ratio stats: mean={r_all.mean():.4f} std={r_all.std():.4f}")
    print(f"    p5={pcts[0]:.4f} p25={pcts[1]:.4f} p50={pcts[2]:.4f} p75={pcts[3]:.4f} p95={pcts[4]:.4f}")
    print(f"    fraction clipped: {100*frac_clipped:.1f}%")

    for adv_name, adv_fn in ADV_FNS.items():
        print(f"\n{'─'*70}")
        print(f"Advantage scheme: {adv_name}")
        print(f"{'─'*70}")

        trajs_with_adv = []
        for group in groups:
            rewards = [t.reward for t in group]
            advs = adv_fn(rewards)
            for traj, adv in zip(group, advs):
                trajs_with_adv.append((traj, adv))

        cats = [t.category for t, _ in trajs_with_adv]
        total_tokens = sum(t.length for t, _ in trajs_with_adv)
        print(f"  Trajectories: {len(trajs_with_adv)} "
              f"(trunc={cats.count('truncated')}, wrong={cats.count('wrong')}, correct={cats.count('correct')})")
        for cat in ["truncated", "wrong", "correct"]:
            ct = sum(t.length for t, _ in trajs_with_adv if t.category == cat)
            advs_cat = [a for t, a in trajs_with_adv if t.category == cat]
            ma = np.mean(advs_cat) if advs_cat else 0
            print(f"  {cat}: {ct} tokens ({100*ct/total_tokens:.0f}%), mean_adv={ma:.4f}")

        for loss_name, loss_fn in LOSS_FNS.items():
            # Use the SAME pre-trained learner model for all loss functions
            # (we just compute different losses, not train further)
            model = deepcopy(learner_model)

            # Total gradient
            model.zero_grad()
            total_loss = torch.tensor(0.0)
            for traj, adv in trajs_with_adv:
                lp = model.log_probs(traj.tokens, traj.actions)
                loss = loss_fn(lp, traj.sampler_lp, torch.full((traj.length,), adv))
                total_loss = total_loss + loss
            total_loss.backward()
            grad_total = torch.cat([p.grad.flatten().clone() for p in model.parameters()])

            # Per-category gradients
            grads_by_cat = {}
            for cat in ["truncated", "wrong", "correct"]:
                model.zero_grad()
                cat_loss = torch.tensor(0.0)
                for traj, adv in trajs_with_adv:
                    if traj.category != cat:
                        continue
                    lp = model.log_probs(traj.tokens, traj.actions)
                    loss = loss_fn(lp, traj.sampler_lp, torch.full((traj.length,), adv))
                    cat_loss = cat_loss + loss
                cat_loss.backward()
                grads_by_cat[cat] = torch.cat([p.grad.flatten().clone() for p in model.parameters()])

            grad_sum = sum(grads_by_cat.values())
            decomp_err = (grad_total - grad_sum).abs().max().item()

            total_l2 = grad_total.norm().item()
            print(f"\n  [{loss_name}]  decomp_err={decomp_err:.2e}  total_grad_L2={total_l2:.2f}")
            print(f"  {'Category':<12} {'L2 norm':>12} {'cos(cat,total)':>16} {'frac of L2':>12}")
            for cat in ["truncated", "wrong", "correct"]:
                g = grads_by_cat[cat]
                l2 = g.norm().item()
                cos = F.cosine_similarity(g.unsqueeze(0), grad_total.unsqueeze(0)).item()
                print(f"  {cat:<12} {l2:12.2f} {cos:16.4f} {l2/total_l2:12.2f}")

            for c1 in ["truncated", "wrong", "correct"]:
                for c2 in ["truncated", "wrong", "correct"]:
                    if c1 >= c2:
                        continue
                    cos_pair = F.cosine_similarity(
                        grads_by_cat[c1].unsqueeze(0), grads_by_cat[c2].unsqueeze(0)).item()
                    print(f"    cos({c1[:5]},{c2[:5]}) = {cos_pair:.4f}", end="")
            print()

            dominant = max(grads_by_cat, key=lambda c: F.cosine_similarity(
                grads_by_cat[c].unsqueeze(0), grad_total.unsqueeze(0)).item())
            print(f"  --> Dominant: {dominant}")

        # Cross-loss gradient cosine
        print(f"\n  Cross-loss gradient cosine ({adv_name}):")
        cross_grads = {}
        for loss_name, loss_fn in LOSS_FNS.items():
            model = deepcopy(learner_model)
            model.zero_grad()
            total_loss = torch.tensor(0.0)
            for traj, adv in trajs_with_adv:
                lp = model.log_probs(traj.tokens, traj.actions)
                loss = loss_fn(lp, traj.sampler_lp, torch.full((traj.length,), adv))
                total_loss = total_loss + loss
            total_loss.backward()
            cross_grads[loss_name] = torch.cat([p.grad.flatten().clone() for p in model.parameters()])

        for l1 in LOSS_FNS:
            for l2 in LOSS_FNS:
                if l1 >= l2:
                    continue
                cos = F.cosine_similarity(
                    cross_grads[l1].unsqueeze(0), cross_grads[l2].unsqueeze(0)).item()
                l2_ratio = cross_grads[l1].norm().item() / cross_grads[l2].norm().item()
                print(f"    cos({l1},{l2}) = {cos:.6f}  L2_ratio = {l2_ratio:.4f}")


# ─────────────────────────────────────────────
# EXPERIMENT 2: Training dynamics (300 steps, off-policy)
#
# Key design: sampler model is updated every K=5 steps (inner loop).
# The learner takes multiple gradient steps on the same batch.
# This creates off-policy conditions where loss functions genuinely differ.
# ─────────────────────────────────────────────

def experiment_2():
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Training dynamics — 300 steps, 12 configs")
    print("  B=8, G=8, max_len=150, sampler_update_freq=5 (off-policy)")
    print("=" * 80)

    EXP2_MAX_LEN = 150
    EXP2_B, EXP2_G = 8, 8
    EXP2_STEPS = 300
    SAMPLER_UPDATE_FREQ = 5  # update sampler every K steps

    configs = []
    for loss_name in ["PPO", "CISPO", "REINFORCE"]:
        for adv_name in ["maxrl", "mean_center"]:
            for agg_name in ["token_sum", "per_traj_mean"]:
                configs.append((loss_name, adv_name, agg_name))

    torch.manual_seed(999)
    ref_tokens = torch.randint(0, VOCAB, (20,))
    ref_actions = torch.randint(0, VOCAB, (20,))

    results = {}
    t_start = time.time()

    for ci, (loss_name, adv_name, agg_name) in enumerate(configs):
        config_key = f"{loss_name}/{adv_name}/{agg_name}"
        print(f"  [{ci+1}/12] {config_key} ...", end="", flush=True)
        t0 = time.time()

        torch.manual_seed(42)
        model = ToyPolicy()
        sampler = ToyPolicy()
        sampler.load_state_dict(model.state_dict())
        for p in sampler.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = LOSS_FNS[loss_name]
        adv_fn = ADV_FNS[adv_name]

        entropy_log, grad_norm_log, correctness_log = [], [], []
        ratio_mean_log = []
        rng = np.random.RandomState(12345)

        current_batch = None

        for step in range(EXP2_STEPS):
            # Re-sample batch every SAMPLER_UPDATE_FREQ steps (simulating rollout refresh)
            if step % SAMPLER_UPDATE_FREQ == 0:
                sampler.load_state_dict(model.state_dict())
                current_batch = generate_batch_with_model(
                    EXP2_B, EXP2_G, rng, sampler, max_len=EXP2_MAX_LEN)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            total_entropy = 0.0
            n_tokens = 0
            total_ratio_sum = 0.0
            total_ratio_n = 0

            for group in current_batch:
                rewards = [t.reward for t in group]
                advs = adv_fn(rewards)
                for traj, adv_val in zip(group, advs):
                    learner_lp = model.log_probs(traj.tokens, traj.actions)

                    adv_eff = adv_val / traj.length if agg_name == "per_traj_mean" else adv_val
                    adv_t = torch.full((traj.length,), adv_eff)
                    loss = loss_fn(learner_lp, traj.sampler_lp, adv_t)
                    total_loss = total_loss + loss

                    with torch.no_grad():
                        ratio = torch.exp(learner_lp - traj.sampler_lp)
                        total_ratio_sum += ratio.sum().item()
                        total_ratio_n += traj.length

                    total_entropy += model.entropy_mean(traj.tokens) * traj.length
                    n_tokens += traj.length

            total_loss.backward()
            gn = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
            optimizer.step()

            entropy_log.append(total_entropy / n_tokens)
            grad_norm_log.append(gn)
            ratio_mean_log.append(total_ratio_sum / total_ratio_n)
            with torch.no_grad():
                correctness_log.append(model.log_probs(ref_tokens, ref_actions).sum().item())

        results[config_key] = {
            "entropy": entropy_log, "grad_norm": grad_norm_log,
            "correctness": correctness_log, "ratio_mean": ratio_mean_log
        }
        elapsed = time.time() - t0
        print(f" {elapsed:.0f}s  ent={entropy_log[-1]:.3f} corr={correctness_log[-1]:.1f} "
              f"ratio_mean={ratio_mean_log[-1]:.3f}")

    total_time = time.time() - t_start
    print(f"\n  Total experiment 2 time: {total_time:.0f}s")

    # Summary table
    print(f"\n{'─'*135}")
    print(f"{'Config':<38} {'Ent_0':>7} {'Ent_100':>8} {'Ent_200':>8} {'Ent_300':>8} "
          f"{'GradNrm_μ':>10} {'Ratio_μ':>8} {'Corr_0':>8} {'Corr_300':>9} {'Status':>12}")
    print(f"{'─'*135}")

    for key, r in results.items():
        ent = r["entropy"]
        gn = r["grad_norm"]
        corr = r["correctness"]
        rm = r["ratio_mean"]
        ent_trend = ent[-1] - ent[0]
        gn_cv = np.std(gn[-50:]) / (np.mean(gn[-50:]) + 1e-8)
        if ent[-1] < 0.5:
            status = "COLLAPSED"
        elif abs(ent_trend) < 0.1 and gn_cv < 0.5:
            status = "converged"
        elif gn_cv > 1.0:
            status = "oscillating"
        elif ent_trend > 0.5:
            status = "diverging"
        else:
            status = "converging"

        print(f"{key:<38} {ent[0]:7.3f} {ent[99]:8.3f} {ent[199]:8.3f} {ent[-1]:8.3f} "
              f"{np.mean(gn):10.1f} {np.mean(rm):8.3f} {corr[0]:8.1f} {corr[-1]:9.1f} {status:>12}")

    # Entropy trajectory detail
    checkpoints = [0, 50, 100, 150, 200, 250, 299]
    print(f"\nEntropy at checkpoints:")
    header = f"{'Step':<6}"
    for k in results:
        short = k.replace("/token_sum", "/sum").replace("/per_traj_mean", "/ptm")
        header += f"  {short:>16}"
    print(header)
    for s in checkpoints:
        row = f"{s:<6}"
        for k in results:
            row += f"  {results[k]['entropy'][s]:16.4f}"
        print(row)

    # Grad norm trajectory
    print(f"\nGrad norm at checkpoints:")
    header = f"{'Step':<6}"
    for k in results:
        short = k.replace("/token_sum", "/sum").replace("/per_traj_mean", "/ptm")
        header += f"  {short:>16}"
    print(header)
    for s in checkpoints:
        row = f"{s:<6}"
        for k in results:
            row += f"  {results[k]['grad_norm'][s]:16.1f}"
        print(row)

    return results


# ─────────────────────────────────────────────
# EXPERIMENT 3: CISPO ratio analysis
# ─────────────────────────────────────────────

def experiment_3():
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CISPO ratio analysis (off-policy)")
    print("=" * 80)

    MAX_LEN = 200

    torch.manual_seed(42)
    sampler_model = ToyPolicy()
    for p in sampler_model.parameters():
        p.requires_grad_(False)

    model = ToyPolicy()
    model.load_state_dict(sampler_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rng = np.random.RandomState(42)

    # Fixed analysis batch
    rng_analysis = np.random.RandomState(999)
    analysis_groups = generate_batch_with_model(16, 8, rng_analysis, sampler_model, max_len=MAX_LEN)

    def analyze_ratios(model, label):
        all_ratios = []
        ratios_by_cat = {"truncated": [], "wrong": [], "correct": []}
        for group in analysis_groups:
            for traj in group:
                with torch.no_grad():
                    l_lp = model.log_probs(traj.tokens, traj.actions)
                    ratio = torch.exp(l_lp - traj.sampler_lp)
                    all_ratios.append(ratio)
                    ratios_by_cat[traj.category].append(ratio)

        r = torch.cat(all_ratios)
        n = len(r)
        n_low = (r < CLIP_LOW).sum().item()
        n_high = (r > CLIP_HIGH).sum().item()
        pcts = np.percentile(r.numpy(), [1, 5, 25, 50, 75, 95, 99])

        print(f"\n  [{label}] {n} tokens")
        print(f"    mean={r.mean():.6f} std={r.std():.6f} min={r.min():.6f} max={r.max():.6f}")
        print(f"    p1={pcts[0]:.4f} p5={pcts[1]:.4f} p25={pcts[2]:.4f} p50={pcts[3]:.4f} "
              f"p75={pcts[4]:.4f} p95={pcts[5]:.4f} p99={pcts[6]:.4f}")
        print(f"    clipped: {n_low} ({100*n_low/n:.1f}%) below, "
              f"{n_high} ({100*n_high/n:.1f}%) above, "
              f"{n-n_low-n_high} ({100*(n-n_low-n_high)/n:.1f}%) in range")

        print(f"    {'Category':<12} {'N':>8} {'Mean':>10} {'Std':>10} {'Median':>10} "
              f"{'%<0.8':>7} {'%>1.28':>7}")
        for cat in ["truncated", "wrong", "correct"]:
            if ratios_by_cat[cat]:
                rc = torch.cat(ratios_by_cat[cat])
                nc = len(rc)
                print(f"    {cat:<12} {nc:8d} {rc.mean():10.6f} {rc.std():10.6f} {rc.median():10.6f} "
                      f"{100*(rc<CLIP_LOW).sum().item()/nc:6.1f}% {100*(rc>CLIP_HIGH).sum().item()/nc:6.1f}%")

    analyze_ratios(model, "Step 0 (on-policy)")

    # Train 10 steps, sampling from init model (off-policy accumulates)
    for step in range(10):
        # Sample fresh batch from STALE sampler each step
        groups = generate_batch_with_model(16, 8, rng, sampler_model, max_len=MAX_LEN)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for group in groups:
            rewards = [t.reward for t in group]
            advs = compute_advantages_mean_center(rewards)
            for traj, adv_val in zip(group, advs):
                lp = model.log_probs(traj.tokens, traj.actions)
                adv_t = torch.full((traj.length,), adv_val)
                total_loss = total_loss + cispo_loss(lp, traj.sampler_lp, adv_t)
        total_loss.backward()
        optimizer.step()

    analyze_ratios(model, "After 10 CISPO steps (vs init sampler)")

    # 50 more steps
    for step in range(50):
        groups = generate_batch_with_model(16, 8, rng, sampler_model, max_len=MAX_LEN)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for group in groups:
            rewards = [t.reward for t in group]
            advs = compute_advantages_mean_center(rewards)
            for traj, adv_val in zip(group, advs):
                lp = model.log_probs(traj.tokens, traj.actions)
                adv_t = torch.full((traj.length,), adv_val)
                total_loss = total_loss + cispo_loss(lp, traj.sampler_lp, adv_t)
        total_loss.backward()
        optimizer.step()

    analyze_ratios(model, "After 60 CISPO steps (vs init sampler)")

    # Compare gradient norms across loss functions at this off-policy state
    print(f"\n  ─── Gradient comparison at step 60 (same model, same batch) ───")
    rng3 = np.random.RandomState(777)
    cmp_groups = generate_batch_with_model(16, 8, rng3, sampler_model, max_len=MAX_LEN)

    cross_grads = {}
    for loss_name, loss_fn in LOSS_FNS.items():
        model_copy = deepcopy(model)
        model_copy.zero_grad()
        total_loss = torch.tensor(0.0)
        n_tok = 0
        for group in cmp_groups:
            rewards = [t.reward for t in group]
            advs = compute_advantages_mean_center(rewards)
            for traj, adv_val in zip(group, advs):
                lp = model_copy.log_probs(traj.tokens, traj.actions)
                adv_t = torch.full((traj.length,), adv_val)
                total_loss = total_loss + loss_fn(lp, traj.sampler_lp, adv_t)
                n_tok += traj.length
        total_loss.backward()
        g = torch.cat([p.grad.flatten() for p in model_copy.parameters()])
        cross_grads[loss_name] = g
        print(f"  {loss_name:<12} grad_L2={g.norm():.2f}")

    print(f"\n  Cross-loss cosine similarity:")
    for l1 in LOSS_FNS:
        for l2 in LOSS_FNS:
            if l1 >= l2:
                continue
            cos = F.cosine_similarity(
                cross_grads[l1].unsqueeze(0), cross_grads[l2].unsqueeze(0)).item()
            print(f"    cos({l1},{l2}) = {cos:.6f}")


# ─────────────────────────────────────────────
# EXPERIMENT 4: Interaction effects
# ─────────────────────────────────────────────

def experiment_4(results_from_exp2: Dict):
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Interaction effects — (loss x advantage x aggregation)")
    print("=" * 80)

    if not results_from_exp2:
        print("  [Skipped]")
        return

    data = {}
    for key, r in results_from_exp2.items():
        parts = key.split("/")
        loss, adv, agg = parts[0], parts[1], parts[2]
        data[(loss, adv, agg)] = {
            "ent": r["entropy"][-1],
            "corr": r["correctness"][-1],
            "gn": np.mean(r["grad_norm"]),
            "ent_delta": r["entropy"][-1] - r["entropy"][0],
        }

    # Full table
    print(f"\n  {'Config':<38} {'Entropy':>9} {'Ent_Δ':>10} {'Correctness':>12} {'GradNorm':>10}")
    print(f"  {'─'*79}")
    for key in sorted(data.keys()):
        d = data[key]
        label = f"{key[0]}/{key[1]}/{key[2]}"
        print(f"  {label:<38} {d['ent']:9.4f} {d['ent_delta']:+10.4f} {d['corr']:12.2f} {d['gn']:10.1f}")

    # Main effects
    print(f"\n  ─── Main effects (averaged) ───")
    for factor, values in [
        ("Loss", ["PPO", "CISPO", "REINFORCE"]),
        ("Adv", ["maxrl", "mean_center"]),
        ("Agg", ["token_sum", "per_traj_mean"]),
    ]:
        print(f"\n  {factor}:")
        for v in values:
            if factor == "Loss":
                cells = [data[(v, a, g)] for a in ["maxrl", "mean_center"]
                         for g in ["token_sum", "per_traj_mean"]]
            elif factor == "Adv":
                cells = [data[(l, v, g)] for l in ["PPO", "CISPO", "REINFORCE"]
                         for g in ["token_sum", "per_traj_mean"]]
            else:
                cells = [data[(l, a, v)] for l in ["PPO", "CISPO", "REINFORCE"]
                         for a in ["maxrl", "mean_center"]]
            me = np.mean([c["ent"] for c in cells])
            mc = np.mean([c["corr"] for c in cells])
            mg = np.mean([c["gn"] for c in cells])
            print(f"    {v:<16} ent={me:.4f}  corr={mc:.2f}  grad={mg:.1f}")

    # Two-way interactions: loss x advantage (entropy)
    print(f"\n  ─── loss x advantage interaction (entropy) ───")
    print(f"  {'':>12} {'maxrl':>10} {'mean_ctr':>10} {'delta':>10}")
    deltas_la_ent = {}
    for loss in ["PPO", "CISPO", "REINFORCE"]:
        e_m = np.mean([data[(loss, "maxrl", g)]["ent"] for g in ["token_sum", "per_traj_mean"]])
        e_c = np.mean([data[(loss, "mean_center", g)]["ent"] for g in ["token_sum", "per_traj_mean"]])
        deltas_la_ent[loss] = e_m - e_c
        print(f"  {loss:<12} {e_m:10.4f} {e_c:10.4f} {e_m-e_c:+10.4f}")
    interaction_la = max(deltas_la_ent.values()) - min(deltas_la_ent.values())
    print(f"  Interaction magnitude: {interaction_la:.4f}")

    # loss x aggregation (entropy)
    print(f"\n  ─── loss x aggregation interaction (entropy) ───")
    print(f"  {'':>12} {'tok_sum':>10} {'ptm':>10} {'delta':>10}")
    deltas_lg_ent = {}
    for loss in ["PPO", "CISPO", "REINFORCE"]:
        e_s = np.mean([data[(loss, a, "token_sum")]["ent"] for a in ["maxrl", "mean_center"]])
        e_p = np.mean([data[(loss, a, "per_traj_mean")]["ent"] for a in ["maxrl", "mean_center"]])
        deltas_lg_ent[loss] = e_s - e_p
        print(f"  {loss:<12} {e_s:10.4f} {e_p:10.4f} {e_s-e_p:+10.4f}")
    interaction_lg = max(deltas_lg_ent.values()) - min(deltas_lg_ent.values())
    print(f"  Interaction magnitude: {interaction_lg:.4f}")

    # loss x advantage (correctness)
    print(f"\n  ─── loss x advantage interaction (correctness) ───")
    print(f"  {'':>12} {'maxrl':>10} {'mean_ctr':>10} {'delta':>10}")
    deltas_la_corr = {}
    for loss in ["PPO", "CISPO", "REINFORCE"]:
        c_m = np.mean([data[(loss, "maxrl", g)]["corr"] for g in ["token_sum", "per_traj_mean"]])
        c_c = np.mean([data[(loss, "mean_center", g)]["corr"] for g in ["token_sum", "per_traj_mean"]])
        deltas_la_corr[loss] = c_m - c_c
        print(f"  {loss:<12} {c_m:10.2f} {c_c:10.2f} {c_m-c_c:+10.2f}")
    print(f"  Interaction magnitude: {max(deltas_la_corr.values()) - min(deltas_la_corr.values()):.2f}")

    # loss x aggregation (correctness)
    print(f"\n  ─── loss x aggregation interaction (correctness) ───")
    print(f"  {'':>12} {'tok_sum':>10} {'ptm':>10} {'delta':>10}")
    deltas_lg_corr = {}
    for loss in ["PPO", "CISPO", "REINFORCE"]:
        c_s = np.mean([data[(loss, a, "token_sum")]["corr"] for a in ["maxrl", "mean_center"]])
        c_p = np.mean([data[(loss, a, "per_traj_mean")]["corr"] for a in ["maxrl", "mean_center"]])
        deltas_lg_corr[loss] = c_s - c_p
        print(f"  {loss:<12} {c_s:10.2f} {c_p:10.2f} {c_s-c_p:+10.2f}")
    print(f"  Interaction magnitude: {max(deltas_lg_corr.values()) - min(deltas_lg_corr.values()):.2f}")

    # Key findings
    print(f"\n  ─── Key findings ───")
    print(f"  Q: Does maxrl amplification interact with loss function choice?")
    print(f"    Interaction magnitude (entropy): {interaction_la:.4f}")
    for loss in ["PPO", "CISPO", "REINFORCE"]:
        print(f"      {loss}: maxrl-mc delta = {deltas_la_ent[loss]:+.4f}")

    print(f"\n  Q: Does per_traj_mean interact with loss function choice?")
    print(f"    Interaction magnitude (entropy): {interaction_lg:.4f}")
    for loss in ["PPO", "CISPO", "REINFORCE"]:
        print(f"      {loss}: sum-ptm delta = {deltas_lg_ent[loss]:+.4f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()
    experiment_1()
    results_exp2 = experiment_2()
    experiment_3()
    experiment_4(results_exp2)
    print(f"\nTotal runtime: {time.time() - t_total:.0f}s")
