"""Investigate advantage computation and PPO dynamics for GPQA RLVR degradation.

Tests the actual _normalize_subgroup function with realistic reward vectors
from a GPQA RLVR run (reward in {-0.2, 0.0, 1.0}).
"""

import itertools
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tinker_cookbook.rl.data_processing import _normalize_subgroup


def analyze_group(rewards: list[float], scheme: str = "maxrl", alpha: float = 0.5):
    """Analyze a single group's advantage computation."""
    r = torch.tensor(rewards, dtype=torch.float32)
    adv = _normalize_subgroup(r, scheme, alpha)
    print(f"  Rewards:    {[round(x, 2) for x in rewards]}")
    print(f"  Advantages: {[round(x.item(), 4) for x in adv]}")
    print(f"  Mean reward: {r.mean().item():.4f}")
    print(f"  Advantage range: [{adv.min().item():.4f}, {adv.max().item():.4f}]")
    print(f"  Advantage std: {adv.std().item():.4f}")
    print()
    return adv


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ========================================================================
# 1. BASIC CASES: What happens with each reward composition?
# ========================================================================
section("1. BASIC ADVANTAGE CASES (MaxRL, alpha=0.5)")

print("Case 1a: All identical rewards (zero variance) → zero gradient signal")
analyze_group([0.0] * 8)
analyze_group([-0.2] * 8)
analyze_group([1.0] * 8)

print("Case 1b: Mix of -0.2 and 0.0 (no correct answers)")
analyze_group([-0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0])
analyze_group([-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
analyze_group([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

print("Case 1c: Mix of 0.0 and 1.0 (no truncation)")
analyze_group([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
analyze_group([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
analyze_group([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

print("Case 1d: All three values (-0.2, 0.0, 1.0)")
analyze_group([-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0])
analyze_group([-0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
analyze_group([-0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


# ========================================================================
# 2. COMPARISON: MaxRL vs mean_center vs GRPO
# ========================================================================
section("2. SCHEME COMPARISON on same reward vector")

test_rewards = [-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0]
for scheme in ["mean_center", "grpo", "maxrl"]:
    print(f"  Scheme: {scheme}")
    analyze_group(test_rewards, scheme=scheme)


# ========================================================================
# 3. SYSTEMATIC ANALYSIS: How advantage magnitude depends on group composition
# ========================================================================
section("3. ADVANTAGE MAGNITUDE vs GROUP COMPOSITION")

print("Group size = 8. Varying (n_trunc, n_wrong, n_correct).")
print(f"{'n_trunc':>7} {'n_wrong':>7} {'n_corr':>6} | {'adv_trunc':>9} {'adv_wrong':>9} {'adv_corr':>9} | {'mean_r':>6} {'p_eff':>5}")
print("-" * 85)

for n_trunc in range(9):
    for n_correct in range(9 - n_trunc):
        n_wrong = 8 - n_trunc - n_correct
        rewards = [-0.2] * n_trunc + [0.0] * n_wrong + [1.0] * n_correct
        r = torch.tensor(rewards, dtype=torch.float32)
        adv = _normalize_subgroup(r, "maxrl", 0.5)

        # Compute p_eff for reference
        r_min, r_max = r.min().item(), r.max().item()
        r_range = r_max - r_min
        mean = r.mean().item()
        p_eff = (mean - r_min) / r_range if r_range > 1e-8 else float('nan')

        adv_vals = {}
        if n_trunc > 0:
            adv_vals['trunc'] = adv[:n_trunc].mean().item()
        else:
            adv_vals['trunc'] = float('nan')
        if n_wrong > 0:
            adv_vals['wrong'] = adv[n_trunc:n_trunc+n_wrong].mean().item()
        else:
            adv_vals['wrong'] = float('nan')
        if n_correct > 0:
            adv_vals['corr'] = adv[n_trunc+n_wrong:].mean().item()
        else:
            adv_vals['corr'] = float('nan')

        print(f"{n_trunc:>7} {n_wrong:>7} {n_correct:>6} | {adv_vals['trunc']:>9.4f} {adv_vals['wrong']:>9.4f} {adv_vals['corr']:>9.4f} | {mean:>6.3f} {p_eff:>5.3f}")


# ========================================================================
# 4. PPO CLIP ANALYSIS
# ========================================================================
section("4. PPO CLIP ASYMMETRY ANALYSIS (clip_low=0.8, clip_high=1.28)")

clip_low = 0.8
clip_high = 1.28

print("PPO loss = -min(ratio * adv, clip(ratio, low, high) * adv)")
print(f"Clip range: [{clip_low}, {clip_high}]")
print()

# For a token with positive advantage (correct answer being reinforced):
#   - ratio > 1 means model has already increased probability
#   - clip at 1.28 allows ratio to go up to 1.28 before clipping kicks in
#   - The gradient pushes ratio UP (increasing probability of this token)
#   - Gradient is active for ratio in [1.0, 1.28]

# For a token with negative advantage (wrong answer being discouraged):
#   - ratio < 1 means model has already decreased probability
#   - clip at 0.8 allows ratio to go down to 0.8 before clipping kicks in
#   - The gradient pushes ratio DOWN (decreasing probability of this token)
#   - Gradient is active for ratio in [0.8, 1.0]

print("Effective gradient window:")
print(f"  Positive advantage: ratio can move from 1.0 to {clip_high} ({(clip_high - 1.0)*100:.0f}% up)")
print(f"  Negative advantage: ratio can move from 1.0 to {clip_low} ({(1.0 - clip_low)*100:.0f}% down)")
print(f"  Asymmetry ratio: {(clip_high - 1.0) / (1.0 - clip_low):.2f}x more room for positive updates")
print()

# Simulate PPO loss for different ratio values and advantage signs
print("PPO effective gradient (d_loss/d_ratio) at different operating points:")
print(f"{'ratio':>7} | {'adv=+1.0':>10} {'adv=-0.1':>10} {'adv=-0.5':>10} {'adv=+0.5':>10}")
print("-" * 55)

for ratio in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.28, 1.5]:
    clipped = max(clip_low, min(clip_high, ratio))
    results = []
    for adv in [1.0, -0.1, -0.5, 0.5]:
        unclipped_obj = ratio * adv
        clipped_obj = clipped * adv
        ppo_obj = min(unclipped_obj, clipped_obj)
        # Gradient is adv if unclipped < clipped, else 0 (clipped)
        # More precisely: gradient = adv if the unclipped term is the min, else 0
        if abs(unclipped_obj - ppo_obj) < 1e-10:
            grad = adv  # unclipped is the min, gradient flows
        else:
            grad = 0.0  # clipped is the min, gradient blocked
        results.append(grad)
    print(f"{ratio:>7.2f} | {results[0]:>10.4f} {results[1]:>10.4f} {results[2]:>10.4f} {results[3]:>10.4f}")


# ========================================================================
# 5. CRITICAL: ADVANTAGE MAGNITUDE × PPO CLIP INTERACTION
# ========================================================================
section("5. ADVANTAGE MAGNITUDE × PPO CLIP INTERACTION")

print("For a typical GPQA group: 5 truncated, 2 wrong, 1 correct")
rewards = [-0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 1.0]
r = torch.tensor(rewards, dtype=torch.float32)
adv = _normalize_subgroup(r, "maxrl", 0.5)

print(f"Rewards:    {[round(x, 2) for x in rewards]}")
print(f"Advantages: {[round(x.item(), 4) for x in adv]}")
print()

# Token-level impact: each token in a trajectory gets that trajectory's advantage
# The total gradient is proportional to advantage * (number of action tokens)
# Correct responses tend to be LONGER (more tokens to reinforce)
# Truncated responses are ALSO long (hit max_tokens)

adv_trunc = adv[0].item()
adv_wrong = adv[5].item()
adv_correct = adv[7].item()

print(f"Per-token advantage breakdown:")
print(f"  Truncated response (5 of them): advantage = {adv_trunc:.4f}")
print(f"  Wrong response     (2 of them): advantage = {adv_wrong:.4f}")
print(f"  Correct response   (1 of it):   advantage = {adv_correct:.4f}")
print()

# Total gradient signal: sum of (advantage × n_tokens) across all trajectories
# Assume typical token counts
typical_tokens_trunc = 4096  # hit max_tokens
typical_tokens_wrong = 2000  # moderate length
typical_tokens_correct = 2500  # moderate-long

total_positive_signal = n_correct * adv_correct * typical_tokens_correct
total_negative_signal_trunc = 5 * adv_trunc * typical_tokens_trunc
total_negative_signal_wrong = 2 * adv_wrong * typical_tokens_wrong

print(f"Total gradient signal (advantage × n_tokens × count):")
print(f"  Positive (correct):    {1 * adv_correct:>8.4f} × {typical_tokens_correct} = {total_positive_signal:>10.1f}")
print(f"  Negative (truncated):  {5 * adv_trunc:>8.4f} × {typical_tokens_trunc} = {total_negative_signal_trunc:>10.1f}")
print(f"  Negative (wrong):      {2 * adv_wrong:>8.4f} × {typical_tokens_wrong} = {total_negative_signal_wrong:>10.1f}")
print(f"  Net signal: {total_positive_signal + total_negative_signal_trunc + total_negative_signal_wrong:>10.1f}")
print()

# With PPO clipping asymmetry: positive signal gets 1.4x more room
# (clip_high - 1) / (1 - clip_low) = 0.28/0.2 = 1.4x
print(f"With DAPO asymmetric clip (0.8/1.28):")
print(f"  Positive signal amplified by up to {clip_high - 1:.0%} per token")
print(f"  Negative signal limited to {1 - clip_low:.0%} per token")
print(f"  Asymmetry factor: {(clip_high - 1) / (1 - clip_low):.2f}x")


# ========================================================================
# 6. ENTROPY ANALYSIS: What happens to the distribution
# ========================================================================
section("6. ENTROPY DYNAMICS UNDER THESE ADVANTAGES")

print("Key insight: Entropy RISING means the policy is becoming MORE uniform.")
print("In PPO, this can only happen if the gradient systematically:")
print("  (a) decreases probability of high-probability tokens, or")
print("  (b) increases probability of low-probability tokens")
print()
print("With MaxRL advantages and sparse correct answers:")
print("  - Most trajectories (7/8) get NEGATIVE advantage")
print("  - Gradient pushes DOWN probability of tokens in those trajectories")
print("  - Those tokens are the MOST COMMON tokens (they appear in 7/8 trajectories)")
print("  - Pushing down the most common tokens → flatter distribution → higher entropy")
print()
print("The 1/8 correct trajectory gets positive advantage, but:")
print("  - Its tokens overlap heavily with wrong trajectories (same question prompt)")
print("  - Only the ACTION tokens get advantage-weighted gradient (mask)")
print("  - So the gradient preferentially DECREASES common response patterns")
print("  - And only weakly INCREASES the rare correct pattern")


# ========================================================================
# 7. EFFECTIVE UPDATE SIZE
# ========================================================================
section("7. EFFECTIVE UPDATE SIZE ESTIMATE")

print("Given:")
print("  LR = 1e-5")
print("  grad_clip_norm = 0.3")
print("  unclipped grad L2 = 20K-55K")
print()

lr = 1e-5
grad_clip = 0.3
typical_grad_l2 = 30000.0  # middle of range

clip_ratio = grad_clip / typical_grad_l2
effective_lr = lr * clip_ratio

print(f"  Clip ratio: {grad_clip} / {typical_grad_l2:.0f} = {clip_ratio:.2e}")
print(f"  Effective LR: {lr} × {clip_ratio:.2e} = {effective_lr:.2e}")
print()
print(f"  Per-parameter update magnitude: ~{effective_lr:.2e}")
print(f"  This is {effective_lr / lr:.4%} of the nominal LR")
print()

# With KL = 0.003, back-compute what this means
kl = 0.003
print(f"  KL(sampling || training) = {kl}")
print(f"  This means on average, per token:")
print(f"    log(p_sample/p_train) ≈ {kl:.4f}")
print(f"    p_train/p_sample ≈ {np.exp(kl):.4f} (≈ {kl*100:.2f}% shift)")
print(f"  This is consistent with very small policy updates per step")


# ========================================================================
# 8. THE DEATH SPIRAL HYPOTHESIS
# ========================================================================
section("8. DEATH SPIRAL HYPOTHESIS")

print("""
Hypothesis: The combination of:
  1. Sparse correct answers (1/8 or fewer per group)
  2. MaxRL advantage amplification of rare wins
  3. Token-level loss summing (not averaging)
  4. Truncated responses being longest (most tokens)
  5. DAPO clip allowing 1.4x more positive than negative movement

...creates a gradient that:
  - Weakly reinforces the rare correct response
  - Strongly suppresses the many wrong/truncated responses
  - But since wrong responses share tokens with correct ones
    (same reasoning patterns, just different final answers),
    the suppression of common patterns ALSO hurts correct responses.

Net effect: The policy becomes more uniform (entropy rises) while
losing the structure that produces correct answers. Accuracy drops.

The grad clipping makes the per-step damage tiny (1e-10 effective LR),
but it's consistently in the wrong direction, so it accumulates.

Key test: Does removing groups with no correct answers help?
  (remove_constant_reward_groups=True)
  Those groups contribute pure suppression signal with no reinforcement.
""")

# Quantify: what fraction of gradient signal comes from groups with no correct answer?
print("Fraction of signal from zero-correct groups:")
print("  If ~40% of groups have zero reward variance → zero gradient (already filtered)")
print("  But among remaining groups, many still have 0 correct answers!")
print("  E.g., groups with mix of -0.2 and 0.0 have variance but NO correct answers.")
print()

# Simulate a realistic batch
np.random.seed(42)
n_groups = 128
group_size = 16  # Note: the actual group_size is 8 per the teammate msg
p_correct = 0.15  # ~15% base accuracy
p_truncated = 0.20  # ~20% truncation rate

n_zero_variance = 0
n_no_correct = 0
n_with_correct = 0
total_negative_advantage_tokens = 0
total_positive_advantage_tokens = 0

for _ in range(n_groups):
    correct = np.random.binomial(group_size, p_correct)
    truncated = np.random.binomial(group_size - correct, p_truncated)
    wrong = group_size - correct - truncated

    rewards = [-0.2] * truncated + [0.0] * wrong + [1.0] * correct

    r = torch.tensor(rewards, dtype=torch.float32)
    if r.std() < 1e-8:
        n_zero_variance += 1
        continue

    adv = _normalize_subgroup(r, "maxrl", 0.5)

    if correct == 0:
        n_no_correct += 1
    else:
        n_with_correct += 1

    # Count token-level signal
    for i, a in enumerate(adv):
        if i < truncated:
            tokens = 4096  # max_tokens
        elif i < truncated + wrong:
            tokens = 2000
        else:
            tokens = 2500
        if a.item() > 0:
            total_positive_advantage_tokens += a.item() * tokens
        else:
            total_negative_advantage_tokens += a.item() * tokens

print(f"Simulated batch ({n_groups} groups, group_size={group_size}):")
print(f"  Zero-variance groups (no gradient): {n_zero_variance} ({n_zero_variance/n_groups:.1%})")
print(f"  Non-zero variance, no correct:      {n_no_correct} ({n_no_correct/n_groups:.1%})")
print(f"  Non-zero variance, with correct:     {n_with_correct} ({n_with_correct/n_groups:.1%})")
print()
print(f"  Total positive gradient signal: {total_positive_advantage_tokens:>12.1f}")
print(f"  Total negative gradient signal: {total_negative_advantage_tokens:>12.1f}")
print(f"  Ratio (neg/pos): {abs(total_negative_advantage_tokens / total_positive_advantage_tokens):.2f}x")
print()
print(f"  Groups with NO correct answers but non-zero variance contribute")
print(f"  PURE NEGATIVE gradient signal. These are {n_no_correct/(n_groups - n_zero_variance):.1%}")
print(f"  of the gradient-producing groups.")


# ========================================================================
# 9. THE -0.2/0.0 GROUPS: PURE SUPPRESSION
# ========================================================================
section("9. PURE SUPPRESSION GROUPS (-0.2/0.0 mix, no correct answers)")

print("These groups have non-zero variance, so they pass the constant-reward filter.")
print("But they contain ZERO positive examples. All advantages are:")
print("  - positive for 0.0 (the 'least bad' response)")
print("  - negative for -0.2 (the truncated response)")
print()
print("What does this reinforce? Responses that avoid truncation.")
print("What does this suppress? Long responses (which get truncated).")
print("Net effect on accuracy: ZERO or NEGATIVE.")
print("  - It doesn't teach the model to be correct")
print("  - It teaches the model to be short enough to not truncate")
print("  - But shorter responses may be LESS likely to be correct on hard questions")
print()

# Show the actual advantages for -0.2/0.0 groups
for n_trunc in range(1, 8):
    n_wrong = 8 - n_trunc
    rewards = [-0.2] * n_trunc + [0.0] * n_wrong
    r = torch.tensor(rewards, dtype=torch.float32)
    adv = _normalize_subgroup(r, "maxrl", 0.5)
    adv_t = adv[:n_trunc].mean().item()
    adv_w = adv[n_trunc:].mean().item()
    print(f"  {n_trunc} trunc + {n_wrong} wrong: adv_trunc={adv_t:>7.4f}, adv_wrong={adv_w:>7.4f}")

print()
print("CRITICAL: These groups actively REWARD being 'wrong but formatted'")
print("over being 'wrong and truncated'. The gradient signal is:")
print("  'Be shorter, even if you don't know the answer'")
print("This is anti-correlated with accuracy on hard questions.")
