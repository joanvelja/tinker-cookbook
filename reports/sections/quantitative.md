# Quantitative Results

## 1. Eval Trajectories

All models evaluated every 10 steps on 664 held-out OmniMath problems at temperature 0.6. Training uses temperature 1.0. Note: prior runs used top_p=1.0 (SDK default) because top_p was not wired through the CLI; now plumbed as eval_top_p=0.95.

### GPT-OSS-20B (75 train steps, 7 eval points)

| Step | P(correct) | P(boxed) | P(c\|b) | Tok/turn | Trunc% | Reward |
|------|-----------|---------|--------|---------|--------|--------|
| 10 | 0.5482 | 0.7425 | 0.7383 | 4161 | 24.7% | 0.4725 |
| 20 | 0.5407 | 0.8855 | 0.6105 | 3185 | 10.1% | 0.5084 |
| 30 | 0.5527 | 0.8705 | 0.6349 | 3423 | 12.1% | 0.5152 |
| 40 | **0.5723** | 0.8976 | 0.6376 | 3240 | 9.5% | **0.5427** |
| 50 | 0.5572 | 0.9036 | 0.6167 | 3011 | 8.6% | 0.5299 |
| 60 | 0.5633 | **0.9187** | 0.6131 | 3039 | 7.5% | 0.5398 |
| 70 | 0.5316 | 0.9051 | 0.5874 | 3016 | 8.6% | 0.5045 |

Best eval accuracy: **57.23%** at step 40. Format compliance peaked at **91.87%** at step 60. Accuracy declined after step 40 despite continued format improvement, indicating overfitting or P(c|b) decay (see decomposition below).

### Qwen3-30B-Instruct (51 train steps, 5 eval points)

| Step | P(correct) | P(boxed) | P(c\|b) | Tok/turn | Trunc% | Reward |
|------|-----------|---------|--------|---------|--------|--------|
| 10 | 0.5316 | 0.6928 | 0.7674 | 5207 | 30.7% | 0.4395 |
| 20 | 0.5467 | 0.7169 | 0.7626 | 5081 | 28.2% | 0.4620 |
| 30 | 0.5708 | 0.7681 | 0.7431 | 4968 | 23.2% | 0.5012 |
| 40 | 0.5828 | 0.8042 | 0.7247 | 4815 | 19.6% | 0.5241 |
| 50 | **0.6160** | **0.8630** | 0.7138 | 4628 | 13.6% | **0.5751** |

Monotonic improvement across all 50 steps. Best eval accuracy: **61.60%** at step 50 (the final eval). All metrics still trending upward; the run was not converged.

### Qwen3-30B-Think (23 train steps, 2 eval points)

| Step | P(correct) | P(boxed) | P(c\|b) | Tok/turn | Trunc% | Reward |
|------|-----------|---------|--------|---------|--------|--------|
| 10 | 0.2530 | 0.2952 | 0.8571 | 7049 | 70.5% | 0.0416 |
| 20 | 0.2666 | 0.3072 | 0.8676 | 7019 | 69.3% | 0.0587 |

Catastrophically low eval accuracy. The think model has the highest P(c|b) of any model (0.8676) but the lowest P(boxed) (0.3072): 70% of eval responses truncate at max_tokens. When it finishes, it's almost always correct. When it doesn't finish, it always fails. Bimodal behavior confirmed by eval group fractions: all_good=0.2666, all_bad=0.7334 at step 20.

## 2. P(correct) = P(boxed) x P(correct|boxed) Decomposition

The headline metric P(correct) decomposes into a product of format compliance and conditional accuracy. This decomposition reveals that the models' learning dynamics are qualitatively different.

### GPT-OSS: Format up, P(c|b) down, net flat

| Step | P(boxed) | P(c\|b) | delta_boxed | delta_P(c\|b) | delta_correct |
|------|---------|--------|------------|--------------|--------------|
| 10 | 0.7425 | 0.7383 | — | — | — |
| 20 | 0.8855 | 0.6105 | +0.1431 | -0.1278 | -0.0075 |
| 30 | 0.8705 | 0.6349 | +0.1280 | -0.1034 | +0.0045 |
| 40 | 0.8976 | 0.6376 | +0.1551 | -0.1008 | +0.0241 |
| 50 | 0.9036 | 0.6167 | +0.1611 | -0.1217 | +0.0090 |
| 60 | 0.9187 | 0.6131 | +0.1762 | -0.1252 | +0.0151 |
| 70 | 0.9051 | 0.5874 | +0.1627 | -0.1510 | -0.0166 |

Format compliance improved +17.6pp (0.7425 to 0.9187) over 60 steps. P(c|b) declined -15.1pp (0.7383 to 0.5874) over the same window. The gains nearly cancel, yielding only +1.5pp net accuracy improvement at step 60.

**Counterfactual:** If the step-60 model had its own P(boxed) of 0.9187 combined with the step-10 P(c|b) of 0.7383, the predicted accuracy would be **0.6783** — a +10.6pp improvement over the actual best of 0.5723. This assumes P(boxed) and P(c|b) are independent, which they are not (the composition effect couples them). The counterfactual is an optimistic upper envelope, not a demonstrated achievable target.

### Qwen3 Instruct: Both axes improving, format dominates

| Step | P(boxed) | P(c\|b) | delta_boxed | delta_P(c\|b) | delta_correct |
|------|---------|--------|------------|--------------|--------------|
| 10 | 0.6928 | 0.7674 | — | — | — |
| 20 | 0.7169 | 0.7626 | +0.0241 | -0.0048 | +0.0151 |
| 30 | 0.7681 | 0.7431 | +0.0753 | -0.0243 | +0.0392 |
| 40 | 0.8042 | 0.7247 | +0.1114 | -0.0427 | +0.0512 |
| 50 | 0.8630 | 0.7138 | +0.1702 | -0.0536 | +0.0843 |

P(boxed) improved +17.0pp while P(c|b) declined only -5.4pp, yielding a net +8.4pp accuracy improvement. The Instruct model's gains are overwhelmingly format-driven: truncation rate dropped from 30.7% to 13.6%. The P(c|b) decline is mild compared to GPT-OSS.

**Counterfactual:** best_boxed (0.8630) x best_P(c|b) (0.7674) = **0.6622**, an optimistic envelope suggesting up to +4.6pp headroom beyond the actual best of 0.6160 (same independence caveat applies).

### Qwen3 Think: High conditional accuracy, catastrophic truncation

With only 2 eval points, decomposition is limited. P(c|b) is 0.8571 and 0.8676 — the highest of any model. The binding constraint is P(boxed) at 0.30 — truncation, rather than math ability. The think model has the best math but the worst format compliance by a factor of 3x.

## 3. Composition Effect

The P(c|b) decline does not necessarily mean the model's math ability deteriorated. A composition effect contributes:

As P(boxed) rises, previously-truncated responses now complete. These newly-completing responses tend to be on harder problems (the ones that caused the model to think too long and truncate in the first place). These harder problems have lower accuracy conditional on completion, diluting the aggregate P(c|b).

Note: this is a composition/selection effect, not Simpson's paradox proper (which requires opposite trends within subgroups). The negative marginal intervals below show that some previously-correct items are also lost — dilution alone does not fully account for the P(c|b) decline.

**Marginal analysis for GPT-OSS** (change in boxed and correct per interval, out of 664 eval problems):

| Interval | New boxed | New correct | Marginal P(c\|b) |
|----------|----------|------------|-----------------|
| 10→20 | +95 | -5 | negative |
| 30→40 | +18 | +13 | 0.722 |
| 50→60 | +10 | +4 | 0.400 |

The step 10→20 interval shows the strongest composition effect: 95 newly-boxed responses but 5 fewer correct answers. The marginal P(c|b) of the newly-boxed responses is below zero: 95 new completions but 5 fewer correct answers total. This means some previously-correct items were lost even as new items entered the boxed pool — the composition effect (dilution by harder problems) is compounded by item-level regression on problems the model previously solved.

**Marginal analysis for Instruct:**

| Interval | New boxed | New correct | Marginal P(c\|b) |
|----------|----------|------------|-----------------|
| 10→20 | +16 | +10 | 0.625 |
| 20→30 | +34 | +16 | 0.471 |
| 30→40 | +24 | +8 | 0.333 |
| 40→50 | +39 | +22 | 0.564 |

The marginal P(c|b) of newly-boxed responses is consistently lower than the incumbent P(c|b) (~0.76), confirming the composition effect. But critically, the marginal P(c|b) is positive (0.33-0.63), meaning the Instruct model genuinely rescues correct answers via reduced truncation, beyond merely surfacing previously-hidden wrong answers. This is why Instruct's P(correct) improves monotonically while GPT-OSS's is flat.

## 4. Train vs. Eval Gap

Training runs at temperature 1.0; eval at 0.6 (top_p=1.0 in prior runs; now wired as 0.95). The temperature gap systematically affects the comparison.

### GPT-OSS

| Step | Train correct | Eval correct | Gap | Train boxed | Eval boxed | Gap |
|------|-------------|-------------|------|------------|-----------|------|
| 10 | 0.3975 | 0.5482 | -0.1507 | 0.6641 | 0.7425 | -0.0784 |
| 20 | 0.4600 | 0.5407 | -0.0807 | 0.8594 | 0.8855 | -0.0262 |
| 40 | 0.4800 | 0.5723 | -0.0923 | 0.8569 | 0.8976 | -0.0407 |
| 60 | 0.5054 | 0.5633 | -0.0579 | 0.8901 | 0.9187 | -0.0285 |
| 70 | 0.4697 | 0.5316 | -0.0619 | 0.9062 | 0.9051 | +0.0011 |

Eval consistently outperforms train by 6-15pp on correct. The gap narrows over training as format compliance converges (at step 70, boxed is essentially identical between train and eval). The residual 6pp gap at step 70 is attributable to temperature: lower eval temperature produces more deterministic (and more correct) outputs.

### Qwen3 Instruct

| Step | Train correct | Eval correct | Gap | Train boxed | Eval boxed | Gap |
|------|-------------|-------------|------|------------|-----------|------|
| 10 | 0.3872 | 0.5316 | -0.1444 | 0.6133 | 0.6928 | -0.0795 |
| 30 | 0.4253 | 0.5708 | -0.1455 | 0.6470 | 0.7681 | -0.1211 |
| 50 | 0.3887 | 0.6160 | -0.2273 | 0.7544 | 0.8630 | -0.1086 |

The eval-train gap **widens** over training for Instruct (from 14pp to 23pp on correct). This is striking: eval accuracy improves +8.4pp while train accuracy is flat (+3.8pp). The explanation: the lower eval temperature (0.6) amplifies format improvements because the Instruct model's primary failure mode is truncation. At temp=0.6, the model generates shorter responses (fewer dead-end spirals), so the format improvement maps more directly to accuracy improvement. At train temp=1.0, the same format improvements are partially offset by higher-variance sampling.

### Qwen3 Think

| Step | Train correct | Eval correct | Gap |
|------|-------------|-------------|------|
| 10 | 0.3521 | 0.2530 | +0.0990 |
| 20 | 0.3628 | 0.2666 | +0.0962 |

Uniquely among the three models, **train outperforms eval** by ~10pp. The think model's bimodal behavior (100% correct or 100% wrong per group) means that at train time with G=16, the frac_mixed is 1.0 (all groups have variation) — the group construction ensures mixed outcomes. At eval time with single rollouts, however, the model enters the bad mode (raw `<think>` text, always truncated) ~70% of the time. The lower eval temperature may actually hurt this model by pushing it more deterministically into one mode or the other.

## 5. Aggregate Training Metrics

### Train correct (5-step rolling averages)

**GPT-OSS:** Monotonic improvement from 0.396 (step 0) to 0.521 (step 74), +12.5pp. The model steadily learns to solve more problems correctly at train temperature.

**Instruct:** Flat. Rolling averages oscillate between 0.37 and 0.40 across all 50 steps. Train accuracy at step 50 (0.389) is barely above step 0 (0.351). The +8.4pp eval improvement is entirely temperature-mediated.

**Think:** Flat. 0.364 (step 0) to 0.402 (step 22), +3.8pp. Noisy, no clear trend.

### Token compression

| Model | Tok/turn (step 0) | Tok/turn (last) | Change |
|-------|------------------|----------------|--------|
| GPT-OSS | 5780 | 3716 | -2064 (-35.7%) |
| Instruct | 6704 | 6069 | -635 (-9.5%) |
| Think | 7491 | 7416 | -74 (-1.0%) |

GPT-OSS is the only model that learned meaningful compression. The 36% token reduction is the primary mechanism behind its format improvement (fewer truncations).

### Entropy

| Model | Entropy (step 0) | Entropy (last) | Change |
|-------|-----------------|---------------|--------|
| GPT-OSS | 0.8413 | 0.6621 | -0.1792 (-21.3%) |
| Instruct | 0.3254 | 0.2917 | -0.0338 (-10.4%) |
| Think | 0.2701 | 0.2248 | -0.0453 (-16.8%) |

GPT-OSS starts at 3x the entropy of Qwen models (0.84 vs 0.27-0.33 nats) and has the most room for RL to shift the policy. See the entropy bottleneck section for analysis of why this matters.

### KL divergence

All three models maintain KL < 0.002 throughout training (KL penalty disabled, KL=0.0). The policy stays close to the reference despite 50-75 steps of PPO updates. This validates the MaxRL finding that PPO clipping + low LR provides sufficient regularization without explicit KL penalty.

| Model | KL (step 0) | KL (last) |
|-------|------------|----------|
| GPT-OSS | 0.001002 | 0.001064 |
| Instruct | 0.001370 | 0.001501 |
| Think | 0.001841 | 0.001885 |
