"""
PPO Loss Aggregation Experiment: Gradient equivalence, training dynamics, and clipping invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

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


def ppo_token_loss(ratio: torch.Tensor, advantage: float):
    clipped = torch.clamp(ratio, CLIP_LOW, CLIP_HIGH)
    return torch.min(ratio * advantage, clipped * advantage)


# ─────────────────────────────────────────────
# Experiment 1: Gradient equivalence
# ─────────────────────────────────────────────

def experiment_1():
    print("=" * 70)
    print("EXPERIMENT 1: Gradient equivalence test")
    print("=" * 70)

    model = ToyPolicy()
    rng = np.random.RandomState(42)

    # 16 trajectories with lengths 500-4000
    trajs = []
    for i in range(16):
        T = rng.randint(500, 4001)
        tokens = torch.randint(0, VOCAB, (T,))
        actions = torch.randint(0, VOCAB, (T,))
        advantage = rng.randn() * 2.0
        sampler_lp = torch.full((T,), -np.log(VOCAB)) + torch.randn(T) * 0.1
        trajs.append(dict(tokens=tokens, actions=actions, advantage=advantage,
                          sampler_lp=sampler_lp, T=T))

    def compute_grads(formulation: str) -> torch.Tensor:
        model.zero_grad()
        total_loss = torch.tensor(0.0)
        for traj in trajs:
            tokens, actions = traj['tokens'], traj['actions']
            A_i, T_i = traj['advantage'], traj['T']
            learner_lp = model.log_probs(tokens, actions)
            ratio = torch.exp(learner_lp - traj['sampler_lp'])

            if formulation == 'A':
                loss_i = -ppo_token_loss(ratio, A_i).sum()
            elif formulation == 'B':
                loss_i = -ppo_token_loss(ratio, A_i / T_i).sum()
            elif formulation == 'C':
                loss_i = -ppo_token_loss(ratio, A_i).mean()
            else:
                raise ValueError
            total_loss = total_loss + loss_i
        total_loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    grad_A = compute_grads('A')
    grad_B = compute_grads('B')
    grad_C = compute_grads('C')

    def compare(n1, g1, n2, g2):
        cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
        l2r = g1.norm().item() / g2.norm().item()
        maxd = (g1 - g2).abs().max().item()
        exact = maxd < 1e-6
        print(f"  {n1} vs {n2}: cos={cos:.10f}  L2_ratio={l2r:.10f}  max_abs_diff={maxd:.2e}  exact={exact}")

    print()
    compare('A', grad_A, 'B', grad_B)
    compare('A', grad_A, 'C', grad_C)
    compare('B', grad_B, 'C', grad_C)
    print(f"\n  Norms: A={grad_A.norm():.6f}, B={grad_B.norm():.6f}, C={grad_C.norm():.6f}")
    print()


# ─────────────────────────────────────────────
# Experiment 2: Training dynamics
# ─────────────────────────────────────────────

def sample_gpqa_traj(rng, scale=1.0) -> Tuple[float, int]:
    """Returns (reward, length). scale < 1 shrinks lengths for speed."""
    u = rng.random()
    if u < 0.35:
        return -0.2, int(4096 * scale)
    elif u < 0.85:
        l = int(np.clip(rng.normal(2500, 800), 200, 4096) * scale)
        return 0.0, max(l, 20)
    else:
        l = int(np.clip(rng.normal(2500, 800), 200, 4096) * scale)
        return 1.0, max(l, 20)


def advantages_maxrl(rewards):
    bl = max(rewards)
    advs = [r - bl for r in rewards]
    s = np.std(advs) + 1e-8
    return [a / s for a in advs]


def advantages_mean_center(rewards):
    bl = np.mean(rewards)
    advs = [r - bl for r in rewards]
    s = np.std(advs) + 1e-8
    return [a / s for a in advs]


def experiment_2():
    print("=" * 70)
    print("EXPERIMENT 2: Training dynamics comparison (300 steps)")
    print("=" * 70)

    B, G = 16, 8
    N_STEPS = 300
    KL_COEFF = 0.01
    SCALE = 0.1  # 10x shorter sequences for speed. Dynamics preserved.

    configs = [
        ("sum+maxrl",        "sum",  "maxrl",       False),
        ("sum+mean_center",  "sum",  "mean_center", False),
        ("ptm+maxrl",        "ptm",  "maxrl",       False),
        ("ptm+mean_center",  "ptm",  "mean_center", False),
        ("ptm+maxrl+KL",     "ptm",  "maxrl",       True),
    ]

    results = {}

    for config_name, agg, adv_scheme, use_kl in configs:
        print(f"\n  Running: {config_name} ...", end="", flush=True)
        torch.manual_seed(42)

        model = ToyPolicy()
        ref_model = ToyPolicy()
        ref_model.load_state_dict(model.state_dict())
        for p in ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed reference for correctness proxy
        ref_tokens = torch.randint(0, VOCAB, (10,))
        ref_actions = torch.randint(0, VOCAB, (10,))

        entropy_traj, grad_norm_traj, correctness_traj = [], [], []
        rng = np.random.RandomState(12345)

        for step in range(N_STEPS):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            total_entropy = 0.0
            n_tok = 0

            for b in range(B):
                # Sample group
                group_r, group_l = [], []
                for g in range(G):
                    r, l = sample_gpqa_traj(rng, SCALE)
                    group_r.append(r)
                    group_l.append(l)

                if adv_scheme == "maxrl":
                    advs = advantages_maxrl(group_r)
                else:
                    advs = advantages_mean_center(group_r)

                for g in range(G):
                    T = group_l[g]
                    tokens = torch.randint(0, VOCAB, (T,))
                    actions = torch.randint(0, VOCAB, (T,))

                    with torch.no_grad():
                        sampler_lp = model.log_probs(tokens, actions).detach()

                    A_i = advs[g]
                    learner_lp = model.log_probs(tokens, actions)
                    ratio = torch.exp(learner_lp - sampler_lp)

                    if agg == "sum":
                        surr = ppo_token_loss(ratio, A_i)
                        loss_i = -surr.sum()
                    else:  # ptm
                        surr = ppo_token_loss(ratio, A_i / T)
                        loss_i = -surr.sum()

                    if use_kl:
                        with torch.no_grad():
                            ref_lp = ref_model.log_probs(tokens, actions)
                            kl = (sampler_lp - ref_lp).mean().item()
                        loss_i = loss_i + KL_COEFF * kl * T

                    total_loss = total_loss + loss_i

                    with torch.no_grad():
                        logits = model.forward(tokens)
                        probs = F.softmax(logits, dim=-1)
                        ent = -(probs * (probs + 1e-10).log()).sum(-1).mean().item()
                        total_entropy += ent * T
                        n_tok += T

            total_loss.backward()
            gn = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
            optimizer.step()

            entropy_traj.append(total_entropy / n_tok)
            grad_norm_traj.append(gn)

            with torch.no_grad():
                cp = model.log_probs(ref_tokens, ref_actions).sum().item()
                correctness_traj.append(cp)

            if (step + 1) % 100 == 0:
                print(f" {step+1}", end="", flush=True)

        results[config_name] = dict(entropy=entropy_traj, grad_norm=grad_norm_traj,
                                    correctness=correctness_traj)
        print(" done")

    # Summary table
    print("\n" + "-" * 100)
    print(f"{'Config':<20} {'Ent_0':>8} {'Ent_end':>8} {'Ent_Δ':>8} "
          f"{'GN_μ':>10} {'GN_σ':>10} "
          f"{'Corr_0':>8} {'Corr_end':>9} {'Corr_Δ':>8}")
    print("-" * 100)
    for name, r in results.items():
        e, gn, c = r['entropy'], r['grad_norm'], r['correctness']
        print(f"{name:<20} {e[0]:8.4f} {e[-1]:8.4f} {e[-1]-e[0]:+8.4f} "
              f"{np.mean(gn):10.4f} {np.std(gn):10.4f} "
              f"{c[0]:8.4f} {c[-1]:9.4f} {c[-1]-c[0]:+8.4f}")

    # Checkpoint trajectories
    cps = [0, 50, 100, 150, 200, 250, 299]
    for metric in ['entropy', 'grad_norm', 'correctness']:
        print(f"\n{metric} at checkpoints:")
        print(f"{'Step':<6}", end="")
        for name in results:
            print(f"{name:>20}", end="")
        print()
        for s in cps:
            print(f"{s:<6}", end="")
            for name in results:
                print(f"{results[name][metric][s]:20.4f}", end="")
            print()
    print()


# ─────────────────────────────────────────────
# Experiment 3: PPO clip invariance
# ─────────────────────────────────────────────

def experiment_3():
    print("=" * 70)
    print("EXPERIMENT 3: PPO clipping invariance under advantage scaling")
    print("=" * 70)

    rng = np.random.RandomState(777)
    N = 1000

    n_violations = 0
    max_rel_error = 0.0
    n_branch_diff = 0

    for i in range(N):
        ratio_val = rng.uniform(0.5, 2.0)
        A_val = rng.uniform(-3, 3)
        T_val = rng.randint(100, 4001)

        ratio = torch.tensor(ratio_val)
        clipped = torch.clamp(ratio, CLIP_LOW, CLIP_HIGH)

        # Raw
        loss_raw = torch.min(ratio * A_val, clipped * A_val).item()
        # Divided
        A_div = A_val / T_val
        loss_div = torch.min(ratio * A_div, clipped * A_div).item()

        expected = loss_raw / T_val
        if abs(expected) < 1e-15:
            rel_err = abs(loss_div - expected)
        else:
            rel_err = abs(loss_div - expected) / abs(expected)

        if rel_err > 1e-6:
            n_violations += 1
        max_rel_error = max(max_rel_error, rel_err)

        # Branch check
        t1r = (ratio * A_val).item()
        t2r = (clipped * A_val).item()
        br = 1 if t1r <= t2r else 2
        t1d = (ratio * A_div).item()
        t2d = (clipped * A_div).item()
        bd = 1 if t1d <= t2d else 2
        if br != bd:
            n_branch_diff += 1

    print(f"\n  Tested {N} random (ratio, A, T) triples")
    print(f"  Loss scaling violations (rel_err > 1e-6): {n_violations}")
    print(f"  Max relative error: {max_rel_error:.2e}")
    print(f"  Branch selection differences: {n_branch_diff}/{N}")
    print(f"  Invariance holds: {n_branch_diff == 0 and n_violations == 0}")
    print()


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
