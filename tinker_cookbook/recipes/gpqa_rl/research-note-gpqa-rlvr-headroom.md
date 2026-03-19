# GPQA Open-Ended RLVR: Direct-Verifier Reference for Debate

Research note from a GPQA RLVR pilot (2026-03-10).

W&B project: `gpqa-rl-smoke`, run: `gpqa_oe-Qwen-Qwen3-30B-A3B-32rank-...`

## Motivation

Debate training on graduate-level science incurs structural overhead: multi-agent coordination, zero-sum gradient dynamics, judge exploitation risk. Before I can quantify those costs, I need a reference: how much can the same model improve on the same questions in the simplest viable RLVR setup?

I trained Qwen3-30B-A3B on GPQA Extended (open-ended) with binary reward from a strong LLM verifier that has access to the ground truth answer. No debate protocol, no multi-turn argumentation, no opponent. The verifier sees the reference answer and decides CORRECT/INCORRECT, in principle a stronger signal than any debate judge, which must decide from arguments alone. This provides a direct-verifier reference curve: what RLVR achieves when the verifier is maximally informed, against which I can compare debate training.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-30B-A3B (MoE, 30B total / 3B active) |
| Dataset | GPQA Extended, open-ended, train split (~490 train / ~54 eval) |
| Verifier | gpt-5-mini, `reasoning_effort=medium`, sees ground truth |
| Batch size | 16 questions |
| Group size | 8 completions per question |
| Max tokens | 4096 |
| LoRA rank | 32 |
| LR | 5.0×10⁻⁴ (auto via `get_lr`) |
| Advantage | MaxRL (centered within each 8-completion group) |
| Loss | Importance sampling (sum over tokens, no clipping) |
| Renderer | `qwen3_disable_thinking` (suppresses Qwen3's `<think>` mode) |
| KL penalty | None |
| Batches | 22 completed (of 100 planned; killed for wallclock) |

Reward: `format_coef * (format - 1) + correct`. In practice: +1.0 for correct + formatted, 0.0 for wrong + formatted, −0.1 for wrong + unformatted (including truncated). The +0.9 case (correct + unformatted) is unreachable by construction: `check_answer` returns False when `<final_answer>` tags are missing, so the grader is never called.

The verifier also has its own cost: 2816+ gpt-5-mini calls over 22 batches (128 per train batch + eval batches). This is embedded in the sampling wallclock and would need to be accounted for in a fair compute comparison with debate.

## Results

### Eval accuracy (n=54 questions, group_size=1)

| Batch | Correct | Format | Mean reward | Tokens/completion |
|-------|---------|--------|-------------|-------------------|
| 0 (pretrained) | 23.4% | 67.2% | 0.202 | 1029 |
| 10 | 29.7% | 96.9% | 0.294 | 1148 |
| 20 | 35.9% | 79.7% | 0.339 | 2117 |

+12.5pp absolute improvement in 20 batches. With only 54 eval questions, the 95% binomial CI on the batch-20 estimate is roughly [24%, 49%], so the direction is clear but the precision is low. The model saw 352 question presentations (2816 rollouts) across 22 batches, less than one epoch over the ~490 training questions. The learning curve was still improving when I killed the run.

### Group composition (train)

| Metric | Batch 0 | Batch 10 | Batch 21 |
|--------|---------|----------|----------|
| frac_all_bad | 0.19 | 0.38 | 0.12 |
| frac_mixed | 0.69 | 0.44 | 0.81 |
| frac_all_good | 0.12 | 0.19 | 0.06 |

By batch 21, 81% of groups contain both correct and incorrect completions for the same question, the regime where GRPO generates meaningful advantage signal. Only 2/16 questions per batch produce zero signal (all-bad).

### Verbosity spiral (train)

The model discovered that longer reasoning chains increase P(correct) and exploited it:

| Batch | Tokens/completion | Format % |
|-------|-------------------|----------|
| 0 | 725 | 97% |
| 5 | 1130 | 98% |
| 10 | 1289 | 97% |
| 15 | 1573 | 94% |
| 21 | 2228 | 74% |

Completions grew 3× over 22 batches. By batch 17+, the longest completions hit the 4096 token cap and truncate before closing the `<final_answer>` tag, causing format failures. The model doesn't self-correct because the reward asymmetry favors verbosity: truncation costs only −0.1 while correctness pays +1.0. For illustration, if being verbose raises P(correct) from 15% to 30% at the cost of 30% truncation, the EV shift is +0.12. The format penalty barely registers. `format_coef=0.1` is too weak to counteract this.

Addressable via higher `format_coef`, higher `max_tokens`, or answer-first prompting. This is a hyperparameter issue; the next run should use `max_tokens=8192` and `format_coef≥0.3` before I compare to debate.

### Compute profile

| | Seconds | % of wallclock |
|---|---------|---------------|
| Total wallclock (22 batches) | 10,400 (2.9h) | 100% |
| Sampling (rollouts + eval + grading) | 9,406 | 90.4% |
| Eval runs | 716 | 6.9% |
| GPU training | 278 | 2.7% |

Sampling dominates, but part of this is self-inflicted. The run used `sampling_max_connections=16` (the rl/train.py default), meaning only 16 of 128 concurrent rollout coroutines could send requests to Tinker at a time. The remaining 112 queued behind an SDK semaphore, effectively serializing into 8 waves. The debate recipe already solved this by deriving `max_connections = batch_size * group_size` (256+ for typical configs). The SDK supports up to 400 concurrent dispatches; we were using 4% of that capacity.

Additionally, the "sampling" bucket here lumps together three things I cannot currently separate: (1) Tinker model generation, (2) gpt-5-mini grader calls, (3) client-side semaphore queueing. I've added per-phase timers (`time/grader_s`, `time/step_excluding_grader_s`) to future runs. The next run also sets `sampling_max_connections = batch_size * group_size`.

Growing completion length makes later batches slower: 458s at batch 0, 944s at batch 20. Later batches also show higher training time (up to 52s / 8.1% at batch 17), making async overlap more valuable than the 2.7% average suggests.

KL from base policy stayed small throughout (KL_v2 ≈ 0.002). The model isn't drifting far from its pretrained distribution despite meaningful accuracy gains.

## Interpreting headroom

The pretrained Qwen3-30B-A3B scores 23.4% on GPQA Extended open-ended. After 20 batches of RLVR (less than one epoch, under 3 hours) it reaches 35.9% (wide CI; see above), with the learning curve still improving when I stopped it.

This curve is my reference for debate training on the same domain. A model trained via debate on GPQA with comparable compute should be measured against it. The gap between debate's accuracy curve and this one quantifies the cost of the debate protocol: multi-agent coordination, zero-sum gradient dynamics, and whatever credit assignment noise the argumentation structure introduces.

This is not a ceiling. Debate could plausibly exceed this reference if multi-agent argumentation surfaces reasoning the model can't reach via solo chain-of-thought. That would be evidence for debate's value proposition beyond scalable oversight. The comparison also isn't task-identical. Direct RLVR trains the model to answer questions; debate trains it to argue persuasively for correct answers against an adversary. The skills overlap but aren't the same.

The verifier itself is imperfect. gpt-5-mini with medium reasoning may miscalibrate on the hardest GPQA questions. This means the reference curve is approximate, and the true "oracle verifier" curve would be higher. For the headroom question, this is conservative: if debate can't match even an imperfect verifier, the protocol overhead is significant.

## Prerequisites before comparison

Two issues artifactually suppressed this run's performance:

1. The verbosity spiral degraded late-training accuracy. Fix: `max_tokens=8192`, `format_coef≥0.3`.
2. `sampling_max_connections=16` throttled rollout throughput to 1/8 of potential. Fix: `sampling_max_connections = batch_size * group_size` (already applied).

Re-run with both fixes to establish a clean reference curve. Then run debate training on the same model × dataset with matched compute budget (22+ batches) and compare eval accuracy curves.
