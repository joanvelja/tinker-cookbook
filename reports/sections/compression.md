# Compression Analysis: GPT-OSS Analysis Channel Evolution

## The Compression Story

GPT-OSS v3b is the only model that demonstrates clear reasoning compression during training. Over 75 steps, average response length drops 36%:

| Step | Tokens/response | Chars (est.) | Format (boxed) | Entropy |
|------|----------------|-------------|----------------|---------|
| 0 | 5,780 | ~23.1k | 55.4% | 0.841 |
| 10 | 5,242 | ~21.0k | 66.4% | 0.765 |
| 20 | 3,942 | ~15.8k | 85.9% | 0.763 |
| 40 | 3,922 | ~15.7k | 85.7% | 0.707 |
| 60 | 3,557 | ~14.2k | 89.0% | 0.717 |
| 74 | 3,716 | ~14.9k | 90.9% | 0.662 |

The compression is front-loaded: 89% of total token reduction (5,780 to 3,942) happens in the first 20 steps. Steps 20-74 contribute only 226 additional tokens of compression. This matches the rapid format-learning phase, where boxed compliance goes from 55% to 86% in those first 20 steps then plateaus near 90%.

## Mechanism: Dead-End Pruning

The experiment log's reasoning quality probes decompose the compression into three mechanisms:

| Mechanism | Contribution | Description |
|-----------|-------------|-------------|
| Format compliance | 60% | Fewer responses hit the truncation ceiling |
| Reduced metacognitive overhead | 25% | Fewer "But/Wait/Actually" self-correction spirals |
| Reasoning efficiency | 15% | Faster commitment to correct approach, less dead-end exploration |

The model is cutting dead-end exploration and low-value throat-clearing, while preserving useful reasoning steps. Self-correction markers ("Wait", "Actually", "But") dropped 35% in absolute count and 20% in density (markers per 1,000 chars) over the first 15 steps.

Transcript comparison illustrates this concretely. At step 0, the problem "119 is a multiple of which number?" produces 3,307 chars with 18 "But" markers of metacognitive spiraling on a trivial factoring question. At step 15, the problem "Find four numbers whose products sum to primes" produces 2,641 chars with 5 markers. The model tries one example (1,2,3,5), confirms it works, and stops.

The step-15 model reasons with less wasted motion.

## Analysis Channel Behavior

GPT-OSS uses Harmony's `<|channel|>analysis` format for its reasoning. The analysis channel is a separate token stream that the model explicitly enters and exits via channel-switching tokens. The final-channel response is where the boxed answer appears.

Analysis channel median length halved during training:

| Window | Analysis channel median (chars) |
|--------|-------------------------------|
| Steps 0-4 | ~19,600 |
| Steps 15-19 | ~9,200 |

Reduction: 53%. The model still uses the analysis channel 100% of the time. It learned to think shorter while continuing to think on every problem. This is the correct behavior: the analysis channel is where useful reasoning happens, and the model should use it, just more efficiently.

## Analysis Channel Death Mode

A persistent pathology: the model sometimes enters the analysis channel and never transitions to the final channel, producing raw Harmony token emissions without ever generating `\boxed{}`. All such responses score 0% correct.

| Window | Death mode rate |
|--------|----------------|
| v3b Steps 0-4 | 31% (10/32 logged) |
| v3b Steps 15-19 | 19% (6/32 logged) |

From the v2 probe data, the rate showed more dramatic oscillation:

| v2 Window | Analysis channel % of all responses |
|-----------|-------------------------------------|
| Iter 0-4 | 10.6% |
| Iter 5-9 | 4.4% |
| Iter 10-14 | 14.8% |
| Iter 20-24 | 8.8% |
| Iter 25-29 | 45.0% |
| Iter 35-39 | 32.1% |
| Iter 40-42 | 25.0% |

The v2 spike at iter 25-29 (45%) coincided with the worst accuracy dip and longest average response lengths. In v3b, the death mode rate declined from ~31% to ~19%, consistent with PPO clipping and lower LR providing more stable dynamics than v2's importance sampling loss.

## Over-Optimization Phase (Steps 40-70)

GPT-OSS eval peaks at step 40 (57.2% correct) then declines:

| Step | Eval correct | Delta from peak |
|------|-------------|----------------|
| 40 | 57.23% | |
| 50 | 55.72% | -1.51 pp |
| 60 | 56.33% | -0.90 pp |
| 70 | 53.16% | -4.07 pp |

27 problems lost from the 664-problem eval set between step 40 and step 70. Meanwhile, train metrics continue improving: train correct goes from 48.0% (step 40) to 52.1% (step 74). The widening train-eval gap is the signature of over-optimization.

Entropy collapse accompanies the decline: 0.841 to 0.662 nats over 75 steps, a 21% reduction. The policy is concentrating probability mass on patterns that work on the training set but fail to generalize.

## Reward Decomposition

How the model improves its reward (step 0 to step 19):

| Source | Delta | % of total reward improvement |
|--------|-------|------------------------------|
| Penalty reduction (format + EOS) | +0.078 | 86% |
| More correct answers | +0.013 | 14% |
| **Total reward delta** | **+0.090** | **100%** |

About 86% of the observed reward increase came from the auxiliary format/completion components of the reward function, and about 14% came from increased correctness. Format boxed improved from 55% to 82%, format EOS from 59% to 90%. The model learned to finish its responses within the token budget and include `\boxed{}`. Actual math improvement (correct rate 39.6% to 46.0%) was secondary. Note this is reward accounting rather than causal attribution: formatting and correctness are behaviorally coupled (better boxing makes correct answers easier to grade), so the true causal breakdown may differ.

This decomposition reveals a reward design issue. With format_coef=0.15 and eos_coef=0.15, the combined format penalty ranges from 0 to -0.30, while the correctness reward ranges from 0 to +1.0. A model that only learns format compliance improves reward by 0.30 per episode. Solving one more problem adds another 1.0. The ratio seems favorable toward correctness, but format improvement is a surface-level behavioral change that dominates the easier early training steps.

## Structural Advantage of Harmony Format

GPT-OSS's analysis/final channel split gives it a structural advantage over Qwen's `<think>` blocks for learning compression. The channel transition (`<|channel|>analysis` to `<|channel|>final`) is a discrete switching point: "stop reasoning, start answering." The model can learn to trigger this transition earlier without changing its reasoning style.

Qwen's `<think>...</think>` has no equivalent clean boundary. The model must learn when to emit `</think>`, which requires predicting "I have enough reasoning to answer now," a higher-order metacognitive judgment. Combined with the 3x entropy gap, this explains why GPT-OSS compresses in 15 steps while Qwen models show minimal response-length reduction over 50+ steps. Instruct tokens: 6,704 to 6,069, a 9% reduction entirely attributable to higher format compliance.
