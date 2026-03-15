# RLVR Experiment Log

## Session: 2026-03-14

### Experiment Matrix

Three model families, testing the "thinking axis" — does visible chain-of-thought
help or hurt RLVR on competition math?

| Config | Model | Renderer | Think mode |
|--------|-------|----------|-----------|
| A | Qwen3-30B-A3B | qwen3 | Visible `<think>` blocks |
| B | Qwen3-30B-A3B | qwen3_disable_thinking | Suppressed (empty `<think>`) |
| C | Qwen3-30B-A3B-Instruct-2507 | qwen3_instruct | Never thinks |
| D | GPT-OSS-20B | gpt_oss_medium_reasoning | Harmony analysis channel |

Dataset: OmniMath-2 (4428 problems, 85/15 train/eval split). LLM grader with
math-aware prompt (gpt-5-mini). MaxRL advantage scheme throughout.

---

## Run History

### v1 (killed after ~10 batches)

**Config:** batch_size=16, group_size=4, LR=5e-4 (auto), loss=importance_sampling,
KL=0, num_substeps=2, remove_constant_groups=True, max_tokens=8192.

**Problems found:**

1. **GPT-OSS collapsed in 19 batches.** Responses devolved into repeating "medium
   medium medium" — the word from the system prompt `"Reasoning: medium"`. KL spiked
   from 0.008 to 0.074. Root cause: no KL penalty + no ratio clipping + high LR +
   system prompt text becoming a degenerate attractor. `num_substeps=2` accelerated
   the drift. `remove_constant_reward_groups` created selection bias toward lucky
   groups.

2. **Qwen3 no-think (Config B) doesn't actually disable thinking.** The
   `qwen3_disable_thinking` renderer prepends empty `<think>\n\n</think>\n\n` as a
   hint, but the base model ignores it and generates `<think>` blocks anyway. These
   consume the token budget, causing truncation. Config B is not a valid non-thinking
   baseline. **Dropped from the experiment matrix.**

3. **Eval set too large.** With eval_frac=0.15 (664 examples), eval took 2-3 hours
   for verbose models — longer than the training between evals. Runs appeared "stuck"
   during eval.

4. **Log dir collision.** Think and no-think runs shared the same model_name, so
   auto-generated log paths collided. Both wrote to the same directory, corrupting
   transcripts.

### v2 (running, ~30-40 batches)

**Config changes:** KL=0.03, group_size=8, eval_frac=0.025 (~110 eval examples),
explicit log_path per run, no-think config dropped.

**Findings so far (from transcript probes):**

#### GPT-OSS v2 — "Getting smarter but longer"

The decomposition tells the story:

```
P(correct) = P(boxed) * P(right|boxed)

Step 10:  56.6% = 85.1% * 66.5%
Step 40:  50.9% = 66.7% * 76.3%
```

Math ability improved +10pp. Format compliance collapsed -18pp. Net: raw accuracy
dropped 6pp, but the model genuinely learned better math — the truncation hides it.

Truncation rate went from 6% (step 20) to 33% (step 40). The model learned "think
more = more correct" but `format_coef=0.1` is too weak to counteract, and
`eos_coef=0.0` provides zero truncation penalty.

**Verbosity spiral mechanism:**
1. Longer reasoning -> higher P(correct) on problems the model can solve
2. Correct answers give +1.0 reward; format failure costs only -0.1
3. RL amplifies "think more" because the reward ratio is 10:1
4. But thinking more -> longer responses -> more hit max_tokens ceiling
5. Truncated responses lose both correct and format reward
6. The model is caught between "think more to be right" and "think less to finish"

No degenerate outputs, no repetition loops, no mode collapse. KL stable at 0.001.
The v2 KL penalty (0.03) prevented the v1-style "medium" collapse. Zero truncations
in the logged subset (the 32 responses per batch all finish), but the FULL batch
has 33% truncation (logged subset is biased toward successful responses via
num_groups_to_log=4).

**Best checkpoint: step 20** (56.2% correct, 92.9% format, 6% truncation).
**Counterfactual:** step-40 model with step-20 format rate would score 70.9%.

Analysis channel (Harmony `<|channel|>analysis` format) usage grew from ~5% to
~30% of responses. All analysis channel responses score 0% correct — it's a
death mode where the model enters raw Harmony token emission and never produces
a `\boxed{}` answer.

#### Qwen3 Instruct v2 — Healthiest run

Eval improving: 55.0% (step 10) -> 58.1% (step 20). Format improving on train:
48.5% -> 59.5%. Correct trending up but noisy.

Dominant failure mode: truncation at 8192 tokens. 28% of responses hit the ceiling.
Successful responses average 2,931 tokens; failed responses average 8,192 (ceiling).
Bimodal: the model either solves efficiently or enters an unbounded reasoning
spiral.

KL drifting from 0.0014 to 0.0152 (10x in 10 steps). The KL penalty at 0.03 coef
produces ~0.0005 effective penalty against rewards of 0.2-0.4 — functionally zero.
The probe recommended increasing KL to 0.3-1.0 or removing it entirely.

#### Qwen3 Think v2 — Bimodal, not learning

Two modes exist:
- Structured think blocks (JSON API format): 98% correct, ~4800 tokens
- Raw `<think>` text (string format): 0% correct, always 8192 tokens (truncated)

Correlation between structured_frac and correct_frac: **r = 0.999**. The model
doesn't "learn to think better" — it either enters the good mode (structured) or
the bad mode (raw). Which mode it enters depends on problem difficulty, not policy
weights.

With group_size=8, all 8 rollouts on the same problem land in the same mode.
All-correct or all-wrong. `remove_constant_reward_groups` drops them. Almost
no gradient signal. 75% of training data wasted.

### v3 (just launched, ~2 batches)

**Config changes aligned with MaxRL paper:**
- LR: 5e-4 -> **1e-5** (50x lower, LoRA-adjusted from MaxRL's 1e-6 full-param)
- Loss: importance_sampling -> **PPO** with clip 0.2/0.2
- KL: 0.03 -> **0.0** (MaxRL explicitly disables KL)
- Grad clip: 0.0 -> **0.3** (match MaxRL)
- Group size: 8 -> **16** (match MaxRL)
- Batch size: 16 -> **128** (closer to MaxRL's 256)
- Eval temperature: 1.0 -> **0.6** (match MaxRL eval config)
- n_batches: 100 -> **500** (but ~150 = 5 epochs is sufficient)
- eos_coef: still **0.0** (mistake — should be 0.3 based on v2 findings)

**Projected runtime:** ~900s/batch. 150 batches (5 epochs) = ~37 hours.
With async + streaming minibatches: ~31 hours.

**Expected improvements over v2:**
- PPO clipping bounds policy movement per step -> slower verbosity spiral
- Lower LR -> slower drift in general
- `parse_success` gate gives truncated responses zero reward (vs v2's -0.1)
- Larger batch (128 prompts) reduces problem-difficulty variance

**Known gap:** `eos_coef=0.0` means the anti-truncation signal is still weak.
v4 should test `eos_coef=0.3`.

---

## Key Lessons Learned

**Verbosity spiral.** The reward structure `correct(0/1) + format_coef*(boxed-1)`
incentivizes longer reasoning because P(correct) grows with chain length. Longer
responses then hit the max_tokens ceiling. The fix: `eos_coef >= 0.3` to penalize
truncation directly. Without it, the only anti-truncation signal is the implicit
-0.1 format penalty, which is 10x weaker than the +1.0 correct reward.

**KL penalty may be unnecessary for RLVR.** GPT-OSS v1 collapsed at KL=0, but
that was compounded by no clipping and 5e-4 LR. MaxRL and DAPO both run KL=0
successfully by relying on PPO clipping + low LR. At KL=0.03 (our v2), the
effective penalty was ~0.0005 against rewards of ~0.3. Functionally zero.

**Use PPO loss, not importance_sampling.** IS loss has no ratio clipping. A single
lucky batch can cause unbounded policy updates. PPO clips to [1-eps, 1+eps].

**LoRA LR for RL: use 1e-5, not 5e-4.** Tinker's `get_lr()` is SL-calibrated.
MaxRL uses 1e-6 full-param, scaling to 1e-5 via the 10x LoRA rule (Thinking
Machines' "LoRA Without Regret"). The 50x gap explains v1/v2 instability.

**`qwen3_disable_thinking` is unreliable.** The empty-think prefill is a hint the
base model ignores. Qwen3-30B-A3B generates `<think>` blocks regardless. Use
Instruct-2507 for genuine non-thinking baselines.

**`remove_constant_reward_groups` can starve gradient signal.** When problem
difficulty is bimodal (all-correct or all-wrong groups), the filter discards most
data. The think model lost 75% of groups this way. Larger group_size or
difficulty-targeted sampling can help.

**Eval set sizing depends on tokens generated, not example count.** A 664-example
eval at 8192 max_tokens takes 2-3 hours for verbose models. 110 examples suffices
for tracking trends.

**Always use explicit `log_path` per run.** Auto-generated paths from model_name
collide when running multiple configs of the same base model.

---

## Hyperparameter Reference

### MaxRL paper (verified from source code)

```
advantage_estimator = maxrl
train_batch_size = 256 (unique prompts)
rollout.n = 16 (group size)
learning_rate = 1e-6 (full-param) -> 1e-5 (LoRA 10x rule)
max_response_length = 4096
temperature = 1.0 (train), 0.6 (eval, top_p=0.95)
KL = 0.0 (use_kl_loss=False, use_kl_in_reward=False)
clip_ratio = 0.2 / 0.2 (symmetric)
grad_clip = 0.3
ppo_epochs = 1
total_epochs = 5
```

Source: github.com/tajwarfahim/maxrl, `qwen3_experiments/run_qwen3_training.sh`

### DAPO (ByteDance, Qwen2.5-32B)

```
clip_low = 0.2, clip_high = 0.28 (asymmetric "clip-higher")
KL = 0 (removed entirely)
prompt_batch_size = 512
rollouts_per_prompt = 16
temperature = 1.0 (train), 1.0 (eval, top_p=0.7)
token-level loss aggregation
dynamic sampling (filter constant-reward groups)
overlong reward shaping (soft penalty near max_tokens)
```

Source: arxiv.org/abs/2503.14476

### DeepSeek-R1

```
questions_per_step = 32
rollouts_per_question = 16 (G=16)
learning_rate = 3e-6
KL = 0.001
max_length = 32768
temperature = 1.0
clip_ratio = 10 (very loose)
```

Source: arxiv.org/pdf/2501.12948

---

## Appendix A: GPT-OSS v2 Probe Results (batch 0-42)

### A.1 Eval trajectory decomposition

| Step | Eval Correct | P(boxed) | P(right\|boxed) | Avg tokens | Truncation % |
|------|-------------|----------|-----------------|-----------|-------------|
| 10   | 56.6%       | 85.1%    | 66.5%           | 3640      | 13.3%       |
| 20   | 56.2%       | 92.9%    | 60.5%           | 2658      | 5.7%        |
| 30   | 54.4%       | 73.2%    | 74.3%           | 4558      | 26.8%       |
| 40   | 50.9%       | 66.7%    | 76.3%           | 4693      | 33.1%       |

The raw eval accuracy dropped 5.7pp, but P(right|boxed) rose 10pp. Truncation
rate tripled from 13% to 33%, masking the math improvement.

Counterfactual: if step-40 had step-20's format rate (92.9%), predicted eval
accuracy would be 76.3% * 92.9% = 70.9%.

### A.2 Train metrics (5-step rolling averages)

| Steps | Correct | Format | Reward | KL |
|-------|---------|--------|--------|----|
| 0-4   | 0.413   | 0.734  | 0.387  | 0.013 |
| 5-9   | 0.391   | 0.823  | 0.374  | 0.017 |
| 10-14 | 0.333   | 0.617  | 0.295  | 0.015 |
| 15-19 | 0.387   | 0.731  | 0.360  | 0.019 |
| 20-24 | 0.357   | 0.767  | 0.334  | 0.017 |
| 25-29 | 0.375   | 0.621  | 0.337  | 0.015 |
| 30-34 | 0.373   | 0.667  | 0.339  | 0.015 |
| 35-39 | 0.452   | 0.660  | 0.418  | 0.014 |
| 40-42 | 0.437   | 0.691  | 0.406  | 0.015 |

KL peaked at ~0.02 then stabilized. No divergence. Train correct is noisy
(min 0.125, max 0.656 per batch) due to problem-difficulty variance at
batch_size=16.

### A.3 Analysis channel (Harmony `<|channel|>analysis`) usage

The model sometimes enters the raw Harmony analysis format instead of using
structured thinking blocks. All analysis channel responses score 0% correct.

| Window | Analysis channel % |
|--------|-------------------|
| iter 0-4 | 10.6% |
| iter 5-9 | 4.4% |
| iter 10-14 | 14.8% |
| iter 20-24 | 8.8% |
| iter 25-29 | 45.0% |
| iter 30-34 | 31.9% |
| iter 35-39 | 32.1% |
| iter 40-42 | 25.0% |

The spike at iter 25-29 coincides with the accuracy dip and the longest average
response lengths. The rate partially recovered but remains elevated at ~25-30%.

### A.4 Response quality comparison (early vs late)

**Iter 5 (early):** 30 responses. 24 correct, 6 wrong. Avg 8,727 chars. Thinking
blocks are substantive: coordinate geometry, reflection arguments, algebraic
manipulation. Wrong answers are on genuinely hard problems (inequality direction,
stereochemistry of proof structures). 2/30 used analysis channel. 3/30 had
within-response repetition (constraint-checking loops).

**Iter 20 (peak):** 32 responses. 27 correct, 5 wrong. Avg 5,389 chars. Tightest
iteration: thinking blocks 37% shorter than iter 5. 84% accuracy. Zero analysis
channel usage. 1/32 with repetition.

**Iter 42 (latest):** 25 responses. 18 correct, 7 wrong. Avg 12,049 chars.
Responses doubled in length vs the peak. 7/25 used analysis channel. 9/25 had
repetition patterns (mostly constraint-checking loops, e.g. `"A + c2 = I"` x16
while solving a cryptarithm). The repetition is functional but wastes tokens.

Wrong answers at iter 42 are format/aggregation errors: listing solutions instead
of counting them, giving the set `{(2014,0,0), (0,2014,0), (0,0,2014)}` when the
answer is `5` (the count). The reasoning is correct; the final extraction step
fails.

### A.5 Repetition and degenerate outputs

Zero degenerate "medium medium medium" loops across all 43 iterations (v2 KL
penalty prevented the v1 collapse pattern). Repetition rate increased from ~20%
(early) to ~45% (late), but all repetition is functional: constraint-checking
loops during case analysis, not degenerate token-level collapse.

No truncations in the logged subset (32 responses per batch all produce EOS).
The 33% truncation rate appears only in the full batch (128 episodes), indicating
the logged subset (num_groups_to_log=4) is biased toward shorter/successful
responses.

---

## Appendix B: Qwen3 Instruct v2 Probe Results (batch 0-29)

### B.1 Eval trajectory

| Step | Eval Correct | Eval Boxed | Eval EOS |
|------|-------------|-----------|---------|
| 10   | 55.0%       | 76.2%     | 76.5%   |
| 20   | 58.1%       | 80.7%     | 80.9%   |

Both eval metrics trending up (+3pp correct, +5pp format between checkpoints).

### B.2 Train metrics trajectory

| Step | Eps | Correct | Boxed | EOS | KL | Entropy | Tok/turn |
|------|-----|---------|-------|-----|----|---------|---------|
| 0 | 32 | 0.344 | 0.500 | 0.531 | 0.0014 | 0.380 | 7556 |
| 5 | 56 | 0.250 | 0.536 | 0.536 | 0.0103 | 0.363 | 6648 |
| 10 | 72 | 0.222 | 0.681 | 0.681 | 0.0176 | 0.307 | 6870 |
| 15 | 64 | 0.234 | 0.500 | 0.500 | 0.0153 | 0.264 | 7164 |
| 20 | 72 | 0.431 | 0.681 | 0.681 | 0.0205 | 0.326 | 6735 |
| 25 | 88 | 0.227 | 0.466 | 0.489 | 0.0182 | 0.312 | 6961 |
| 29 | 80 | 0.237 | 0.425 | 0.425 | 0.0226 | 0.302 | 7354 |

KL drifting monotonically from 0.001 to 0.023 with no plateau in sight. The KL
penalty at coef=0.03 produces ~0.0007 effective penalty against rewards of 0.2-0.4.
Functionally zero.

### B.3 Truncation analysis

Boxed and EOS are near-identical in every row (boxed ~= EOS, differing by <2pp).
This means failure is almost always hard truncation at 8192 tokens, not wrong
format. The model either finishes (both boxed+EOS) or runs out of tokens (neither).

By grade across 320 logged episodes:
- correct: 190/320 (59.4%), avg 2,931 tokens
- incorrect: 41/320 (12.8%), avg 6,331 tokens
- error (truncated): 89/320 (27.8%), avg 8,192 tokens (ceiling)

The distribution is bimodal: efficient solutions under 3k tokens, or unbounded
reasoning spirals that hit 8192. No middle ground.

### B.4 Qualitative observations

Successful responses use clean step-by-step markdown with `\boxed{}` at the end.
The model handles fraction-type problems, basic combinatorics, and algebraic
manipulation well. Shortest correct response: 116 tokens (a trivial doors problem).

Failed responses are truncated mid-computation. The model attempts multi-digit
arithmetic in text (`94,143,178 * 27 = ...`) and runs out of budget. These are
problems that require a calculator, not more thinking. The model has no mechanism
to detect "this approach will exceed the budget" and bail out with a best guess.

Response style has not meaningfully changed between early and late training.
The model was already verbose at step 0 (avg 7,556 tokens). RL has not taught
it conciseness. The only structural change: slightly higher format compliance
(50% to 60% boxed) as the model learned to place `\boxed{}` more reliably when
it does finish.
