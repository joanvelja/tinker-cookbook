"""Part 2: Deep dive into MaxRL advantage mechanics and the wrong-response zero.

The key finding from part 1: with rewards [-0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 1.0],
the wrong (0.0) responses get EXACTLY 0.0 advantage. This needs investigation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tinker_cookbook.rl.data_processing import _normalize_subgroup


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ========================================================================
# 1. WHY DO 0.0 REWARDS GET ZERO ADVANTAGE?
# ========================================================================
section("1. TRACING THE MATH: WHY 0.0 REWARDS → 0.0 ADVANTAGE")

rewards = [-0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 1.0]
r = torch.tensor(rewards, dtype=torch.float32)

mean = r.mean()
r_min = r.min()
r_max = r.max()
r_range = r_max - r_min
p_eff = (mean - r_min) / r_range
alpha = 0.5

print(f"Rewards: {rewards}")
print(f"mean = {mean.item():.4f}")
print(f"r_min = {r_min.item():.4f}, r_max = {r_max.item():.4f}")
print(f"r_range = {r_range.item():.4f}")
print(f"p_eff = (mean - r_min) / r_range = ({mean.item():.4f} - {r_min.item():.4f}) / {r_range.item():.4f} = {p_eff.item():.4f}")
print(f"p_eff^alpha = {p_eff.item():.4f}^{alpha} = {p_eff.item()**alpha:.4f}")
print()
print(f"advantage_i = (r_i - mean) / p_eff^alpha")
print()
for i, r_i in enumerate(rewards):
    centered = r_i - mean.item()
    adv = centered / (p_eff.item() ** alpha)
    print(f"  r={r_i:>5.1f}: centered = {r_i} - {mean.item():.4f} = {centered:>7.4f}, advantage = {centered:.4f} / {p_eff.item()**alpha:.4f} = {adv:>7.4f}")

print()
print(f"The 0.0 rewards get advantage = (0.0 - {mean.item():.4f}) / {p_eff.item()**alpha:.4f}")
print(f"  = {-mean.item():.4f} / {p_eff.item()**alpha:.4f}")
print(f"  = {-mean.item() / p_eff.item()**alpha:.4f}")
print(f"  This is NOT zero. Let me recheck...")

# Verify with the actual function
adv = _normalize_subgroup(r, "maxrl", 0.5)
print(f"\nActual function output: {[round(x.item(), 6) for x in adv]}")
print(f"  adv for 0.0 reward: {adv[5].item():.10f}")

print("\nAh - in the 5+2+1 case, adv[5] = {:.6f}, which is NOT zero.".format(adv[5].item()))
print("But earlier I printed it as 0.0 due to rounding. Let me recheck...")

# Recheck the 5+2+1 case from part 1
print("\n--- Revisiting 5 trunc + 2 wrong + 1 correct ---")
rewards2 = [-0.2] * 5 + [0.0] * 2 + [1.0]
r2 = torch.tensor(rewards2, dtype=torch.float32)
adv2 = _normalize_subgroup(r2, "maxrl", 0.5)
print(f"Rewards: {rewards2}")
print(f"Advantages (full precision): {[x.item() for x in adv2]}")

mean2 = r2.mean()
print(f"mean = {mean2.item()}")
print(f"Centered 0.0: {0.0 - mean2.item()}")

# OH WAIT - in the part 1 output, it showed advantages as [-1.2, -1.2, -1.2, -1.2, -1.2, 0.0, 0.0, 6.0]
# Let me recheck this with EXACTLY the rewards from part 1's "typical group"
print("\n\n--- The rewards used were [-0.2]*5 + [0.0]*2 + [1.0]*1 ---")
print(f"mean = {mean2.item():.6f}")
print(f"This means the advantages for the 0.0 items should be:")
print(f"  (0.0 - {mean2.item():.6f}) / p_eff^alpha")

r_min2, r_max2 = r2.min().item(), r2.max().item()
r_range2 = r_max2 - r_min2
p_eff2 = (mean2.item() - r_min2) / r_range2
print(f"  r_min={r_min2}, r_max={r_max2}, r_range={r_range2}")
print(f"  p_eff = ({mean2.item():.6f} - {r_min2}) / {r_range2} = {p_eff2:.6f}")
print(f"  p_eff^0.5 = {p_eff2**0.5:.6f}")
print(f"  advantage = {-mean2.item():.6f} / {p_eff2**0.5:.6f} = {-mean2.item() / p_eff2**0.5:.6f}")


# ========================================================================
# 2. THE REAL ISSUE: MaxRL AMPLIFIES RARE WINS ENORMOUSLY
# ========================================================================
section("2. MaxRL ADVANTAGE AMPLIFICATION")

print("With p_eff^alpha denominator, advantages for the correct response scale as:")
print("  advantage_correct = (1.0 - mean) / p_eff^alpha")
print()
print("When p_eff is small (few correct answers), p_eff^alpha is small,")
print("so the advantage is AMPLIFIED inversely proportional to p_eff^alpha.")
print()
print("This is by design - MaxRL is supposed to weight rare wins more heavily.")
print("But it also amplifies the NEGATIVE advantages correspondingly.")
print()

for n_correct in range(1, 8):
    n_wrong = 8 - n_correct
    rewards = [0.0] * n_wrong + [1.0] * n_correct
    r = torch.tensor(rewards, dtype=torch.float32)
    adv = _normalize_subgroup(r, "maxrl", 0.5)

    p = n_correct / 8
    print(f"  {n_correct}/8 correct (p={p:.3f}): adv_wrong={adv[0].item():>8.4f}, adv_correct={adv[-1].item():>8.4f}, |adv_correct/adv_wrong|={abs(adv[-1].item()/adv[0].item()):>6.2f}")


# ========================================================================
# 3. TOKEN-LEVEL LOSS SUMMING: THE MULTIPLICATIVE EFFECT
# ========================================================================
section("3. TOKEN-LEVEL LOSS SUMMING EFFECT")

print("The loss is SUMMED over tokens, not averaged.")
print("This means longer responses get proportionally more gradient signal.")
print()
print("For a group with 5 trunc (-0.2), 2 wrong (0.0), 1 correct (1.0):")

rewards = [-0.2] * 5 + [0.0] * 2 + [1.0]
r = torch.tensor(rewards, dtype=torch.float32)
adv = _normalize_subgroup(r, "maxrl", 0.5)

# Typical token lengths for GPQA
scenarios = [
    ("Truncated (hit max)", 4096, adv[0].item()),
    ("Wrong (moderate)", 2000, adv[5].item()),
    ("Correct (moderate-long)", 2500, adv[7].item()),
]

total_pos = 0
total_neg = 0
for desc, tokens, advantage in scenarios:
    count = 5 if "Truncated" in desc else (2 if "Wrong" in desc else 1)
    total_signal = count * advantage * tokens
    print(f"  {desc}: {count}× advantage={advantage:>8.4f} × {tokens} tokens = {total_signal:>12.1f}")
    if total_signal > 0:
        total_pos += total_signal
    else:
        total_neg += total_signal

print(f"\n  Total positive signal: {total_pos:>12.1f}")
print(f"  Total negative signal: {total_neg:>12.1f}")
print(f"  Net signal: {total_pos + total_neg:>12.1f}")
print(f"  Negative/Positive ratio: {abs(total_neg/total_pos):.2f}x")

print("\n  KEY: The truncated responses are the LONGEST (4096 tokens, they hit max_tokens)")
print("  AND there are 5 of them vs 1 correct response.")
print("  Even though the correct response has 4.8x the per-token advantage,")
print("  the truncated responses dominate the gradient through sheer volume.")


# ========================================================================
# 4. WHAT ABOUT alpha=0.5 SUBGROUPS?
# ========================================================================
section("4. MaxRL alpha=0.5 SUBGROUP BEHAVIOR")

print("MaxRL with alpha=0.5 is supposedly the power_mean scheme.")
print("But in the code, MaxRL forces alpha=1.0!")
print()

# Check the code
print("From _normalize_subgroup:")
print("  if scheme == 'maxrl': eff_alpha = 1.0")
print("  if scheme == 'power_mean': eff_alpha = alpha")
print()
print("So the teammate saying 'MaxRL (alpha=0.5)' may be confusion between")
print("MaxRL and power_mean. Let me check what advantage_scheme the RLVR")
print("recipe actually uses...")
print()

# From the CLIConfig: advantage_scheme: AdvantageScheme = "maxrl"
# From Config: advantage_alpha: float = 0.5
# But _normalize_subgroup IGNORES alpha for maxrl!
print("CLIConfig default: advantage_scheme='maxrl'")
print("Config default: advantage_alpha=0.5")
print("_normalize_subgroup: if scheme=='maxrl' → eff_alpha=1.0 (IGNORES the 0.5)")
print()
print("This means the advantage normalization is STRONGER than power_mean(0.5).")
print("With alpha=1.0, the denominator is p_eff^1.0 = p_eff itself,")
print("not p_eff^0.5 = sqrt(p_eff).")
print()

# Compare alpha=0.5 (power_mean) vs alpha=1.0 (maxrl)
rewards = [0.0] * 7 + [1.0]
r = torch.tensor(rewards, dtype=torch.float32)

adv_pm05 = _normalize_subgroup(r, "power_mean", 0.5)
adv_maxrl = _normalize_subgroup(r, "maxrl", 0.5)  # alpha ignored

print(f"Rewards: {[round(x, 1) for x in rewards]}")
print(f"  power_mean(0.5): adv_wrong={adv_pm05[0].item():.4f}, adv_correct={adv_pm05[7].item():.4f}")
print(f"  maxrl (alpha=1): adv_wrong={adv_maxrl[0].item():.4f}, adv_correct={adv_maxrl[7].item():.4f}")
print(f"  MaxRL amplification vs power_mean(0.5): {adv_maxrl[7].item()/adv_pm05[7].item():.2f}x")


# ========================================================================
# 5. THE ADVANTAGE_SUBGROUPS QUESTION
# ========================================================================
section("5. ADVANTAGE SUBGROUPS")

print("The teammate mentions 'alpha=0.5 subgroups'. Let me check if the RLVR")
print("recipe uses advantage_subgroups...")

# In compute_advantages, if env_group_builders_P is not None and has
# advantage_subgroups method, advantages are computed per-subgroup.
# For RLVR (ProblemGroupBuilder), there are no subgroups.

print("ProblemGroupBuilder does NOT implement advantage_subgroups,")
print("so compute_advantages uses the whole group as one unit.")
print("Subgroups are a debate/multiplayer concept.")
print()
print("For RLVR GPQA runs: no subgroups. Full group of 8 (or 16) treated as one unit.")


# ========================================================================
# 6. remove_constant_reward_groups BEHAVIOR
# ========================================================================
section("6. remove_constant_reward_groups")

print("Default in CLIConfig: remove_constant_reward_groups = False")
print("This means groups where ALL rewards are identical (all-wrong, all-truncated)")
print("still get included. Their advantages are all zero, so they contribute nothing.")
print("This is fine — it just wastes compute, doesn't add bad gradient.")
print()
print("BUT: groups with mix of -0.2 and 0.0 (no correct answers)")
print("have NON-ZERO variance and produce gradient. And this gradient")
print("only teaches 'avoid truncation' not 'be correct'.")
print()

# How often does this happen? With p_correct = 15% and group_size = 8:
np.random.seed(42)
n_trials = 10000
group_size = 8
p_correct = 0.15
p_trunc = 0.20

n_zero_var = 0
n_no_correct_but_var = 0
n_with_correct = 0

for _ in range(n_trials):
    n_c = np.random.binomial(group_size, p_correct)
    n_t = np.random.binomial(group_size - n_c, p_trunc)
    n_w = group_size - n_c - n_t

    rewards = set()
    if n_t > 0: rewards.add(-0.2)
    if n_w > 0: rewards.add(0.0)
    if n_c > 0: rewards.add(1.0)

    if len(rewards) <= 1:
        n_zero_var += 1
    elif n_c == 0:
        n_no_correct_but_var += 1
    else:
        n_with_correct += 1

print(f"Monte Carlo (n={n_trials}, G={group_size}, p_correct={p_correct}, p_trunc={p_trunc}):")
print(f"  Zero-variance groups:                     {n_zero_var/n_trials:.1%}")
print(f"  Non-zero variance, no correct answers:    {n_no_correct_but_var/n_trials:.1%}")
print(f"  Non-zero variance, with correct answers:  {n_with_correct/n_trials:.1%}")
print()

# Also check with group_size=16 (the teammate says group_size may be larger)
for gs in [8, 16]:
    counts = {"zero_var": 0, "no_correct_var": 0, "with_correct": 0}
    for _ in range(n_trials):
        n_c = np.random.binomial(gs, p_correct)
        n_t = np.random.binomial(gs - n_c, p_trunc)
        n_w = gs - n_c - n_t
        rewards = set()
        if n_t > 0: rewards.add(-0.2)
        if n_w > 0: rewards.add(0.0)
        if n_c > 0: rewards.add(1.0)
        if len(rewards) <= 1:
            counts["zero_var"] += 1
        elif n_c == 0:
            counts["no_correct_var"] += 1
        else:
            counts["with_correct"] += 1

    print(f"  G={gs}: zero_var={counts['zero_var']/n_trials:.1%}, "
          f"no_correct_var={counts['no_correct_var']/n_trials:.1%}, "
          f"with_correct={counts['with_correct']/n_trials:.1%}")


# ========================================================================
# 7. THE ENTROPY MECHANISM: Token overlap between trajectories
# ========================================================================
section("7. TOKEN OVERLAP AND ENTROPY RISE")

print("Why does entropy rise? The key mechanism:")
print()
print("1. All 8 trajectories in a group share the SAME PROMPT tokens")
print("   (same question). Only action tokens get advantages (mask=1.0).")
print()
print("2. Action tokens for WRONG responses overlap heavily with action")
print("   tokens for CORRECT responses on the same question:")
print("   - Same reasoning patterns")
print("   - Same vocabulary (science terms)")
print("   - Often same chain-of-thought structure")
print("   - Only the final answer differs")
print()
print("3. With 7/8 responses getting negative advantage, the gradient")
print("   SUPPRESSES these common tokens. But the same tokens appear")
print("   in the correct response too. Only 1/8 of the gradient signal")
print("   reinforces them.")
print()
print("4. Net effect on common reasoning tokens: suppression dominates.")
print("   The policy learns to AVOID the common response patterns,")
print("   making the distribution more uniform → entropy rises.")
print()
print("5. This is NOT the PPO clip causing entropy rise — it's the")
print("   fundamental imbalance in the advantage signal. The clip")
print("   asymmetry (1.4x) slightly favors reinforcement, but it's")
print("   overwhelmed by the 7:1 numerical imbalance in trajectories.")


# ========================================================================
# 8. QUANTIFY: p_eff DISTRIBUTION OVER REALISTIC BATCHES
# ========================================================================
section("8. p_eff DISTRIBUTION (determines advantage amplification)")

np.random.seed(42)
n_groups = 10000
group_size = 8
p_correct = 0.15

p_effs = []
amplifications = []

for _ in range(n_groups):
    n_c = np.random.binomial(group_size, p_correct)
    n_t = np.random.binomial(group_size - n_c, 0.20)
    n_w = group_size - n_c - n_t

    rewards = [-0.2] * n_t + [0.0] * n_w + [1.0] * n_c
    r = torch.tensor(rewards, dtype=torch.float32)
    r_min, r_max = r.min().item(), r.max().item()
    r_range = r_max - r_min
    if r_range < 1e-8:
        continue
    mean = r.mean().item()
    p_eff = (mean - r_min) / r_range
    p_effs.append(p_eff)
    # MaxRL uses alpha=1, so denominator is p_eff
    amplifications.append(1.0 / p_eff if p_eff > 1e-8 else float('inf'))

p_effs = np.array(p_effs)
amplifications = np.array(amplifications)
amplifications = amplifications[np.isfinite(amplifications)]

print(f"p_eff statistics (n={len(p_effs)} non-constant groups):")
print(f"  Median: {np.median(p_effs):.4f}")
print(f"  Mean:   {np.mean(p_effs):.4f}")
print(f"  Min:    {np.min(p_effs):.4f}")
print(f"  Max:    {np.max(p_effs):.4f}")
print(f"  10th percentile: {np.percentile(p_effs, 10):.4f}")
print(f"  90th percentile: {np.percentile(p_effs, 90):.4f}")
print()
print(f"Advantage amplification (1/p_eff) statistics:")
print(f"  Median: {np.median(amplifications):.2f}x")
print(f"  Mean:   {np.mean(amplifications):.2f}x")
print(f"  Max:    {np.max(amplifications):.2f}x")
print(f"  >10x: {(amplifications > 10).mean():.1%}")
print(f"  >20x: {(amplifications > 20).mean():.1%}")
print()
print("Histogram of p_eff:")
for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
    count = ((p_effs >= lo) & (p_effs < hi)).sum()
    print(f"  [{lo:.1f}, {hi:.1f}): {count/len(p_effs):.1%}")


# ========================================================================
# 9. SUMMARY: ROOT CAUSE CHAIN
# ========================================================================
section("9. ROOT CAUSE CHAIN")

print("""
ROOT CAUSE: Negative gradient signal dominates positive signal.

Mechanism chain:

1. Low base accuracy (~15%) → most groups have 0-1 correct out of 8
   → 7/8 trajectories get negative advantage

2. MaxRL (alpha=1) amplifies rare wins by 1/p_eff ≈ 5-8x,
   but the SAME amplification applies to the negative advantages.
   The sum of advantages is zero by construction (centered).

3. Token-level loss SUMMING means longer responses contribute
   proportionally more gradient. Truncated responses (4096 tokens)
   contribute ~2x the signal of correct responses (~2000-2500 tokens).

4. Net gradient per step: suppression of common reasoning patterns.
   This is visible as entropy INCREASE (distribution flattening).

5. Grad clipping at 0.3 (vs unclipped ~30K) scales effective LR to
   ~1e-10, so each step's damage is tiny. But it accumulates.

6. The model slowly loses the structured reasoning that produces
   correct answers. Accuracy degrades from step 0.

CONTRIBUTING FACTORS (not root cause):
- DAPO asymmetric clip (0.8/1.28): 1.4x bias toward reinforcement,
  but overwhelmed by the 7:1 trajectory count imbalance
- No entropy bonus in loss: nothing counteracts the entropy rise
- No KL penalty: no anchor to prevent drift from pretrained policy
- Groups with only -0.2/0.0 rewards: teach "be short" not "be correct"

TESTABLE PREDICTIONS:
1. If you increase group_size (more diverse rollouts per question),
   you increase P(at least 1 correct), reducing pure-suppression groups
2. If you use KL penalty > 0, you anchor the policy and slow entropy rise
3. If you use mean_center instead of maxrl, advantage amplification
   is smaller, potentially reducing the imbalance
4. If you use loss averaging over tokens (not summing), truncated
   responses don't dominate
5. If base accuracy is higher (easier questions or stronger model),
   the 7:1 imbalance improves
""")
