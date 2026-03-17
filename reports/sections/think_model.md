# Qwen3 Think Model: Token Budget, Bimodality, Eval Temperature

## Summary of Outcomes

Qwen3-30B-A3B (think mode) is the worst-performing model in the v3b experiment by a wide margin.

| Metric | Think | Instruct | GPT-OSS |
|--------|-------|----------|---------|
| Eval correct (step 10) | 25.3% | 53.2% | 54.8% |
| Eval correct (latest) | 26.7% (step 20) | 61.6% (step 50) | 53.2% (step 70) |
| Train correct (step 0) | 36.4% | 35.1% | 39.6% |
| Truncation rate (step 0) | ~56% | ~44% | ~41% |
| Entropy (step 0) | 0.27 nats | 0.33 nats | 0.84 nats |

Train correct (36.4%) is comparable to the others. Eval correct (25.3%) is less than half. Two compounding problems: most responses are truncated, losing both answer and format reward; and the eval temperature setting (T=0.6) collapses the bimodal distribution into the wrong mode.

## Bimodal Response Structure

From v2 transcript probes with detailed per-response breakdowns, the think model produces two distinct response types.

Structured think blocks (JSON API format) achieve 98% correct at ~4,800 tokens. The model enters the Qwen3 think API format with organized reasoning steps and reliably produces `\boxed{}` answers.

Raw `<think>` text (string format) achieves 0% correct at exactly 8,192 tokens (truncated). The model enters an unstructured stream-of-consciousness mode that never terminates and never produces a boxed answer.

The correlation between `structured_frac` and `correct_frac` across batches is r = 0.999. The model doesn't learn to think better. It either enters the good mode or the bad mode. Which mode it enters depends on problem properties, and RL training has not shifted this distribution.

With group_size=16, all 16 rollouts on the same problem tend to land in the same mode. Groups are either all-correct or all-wrong, producing zero within-group advantage variance. `remove_constant_reward_groups=True` discards these groups, and the model loses the majority of its gradient signal.

## Think Block Internal Structure

For responses that do produce correct answers, the think block is internally wasteful.

Median position of first correct `\boxed{}`: past 95% of the think block. The model finds the answer near the very end of its reasoning. Post-answer rumination is growing over training (step 6 shows the highest rates across the 7 steps observed). RL has not taught the model to commit to answers faster.

Post-answer activity falls into four categories. Verification ("let me check once more") is sometimes useful. Alternative approaches ("alternatively, using coordinates...") are pure waste, exploring a second method after already solving. Hedging ("wait, but maybe I need to reconsider...") almost always confirms the original answer. Format anxiety ("should I write `\boxed{(a,b)}` or two separate boxes?") is the most egregious: one response burned 10,423 characters deliberating between answer formats.

## The 40% Viable-Path Truncation Problem

~56% of all think model responses are truncated at 8,192 tokens. Of these:

- 40% were on viable paths. The model had found or was converging on the correct answer but ran out of token budget. Pure losses.
- 60% were genuine dead ends: algebraic loops, case enumeration explosions, or unbounded exploration that would not have terminated at any reasonable budget.

The 40% viable-path rate means roughly 22% of all responses (40% of ~56%) have the correct answer somewhere in the think block but receive zero reward because they hit `max_tokens` before producing `\boxed{}`.

Counterfactual scenario analysis: if all viable-path truncations were rescued and scored correct, eval accuracy would reach approximately 26.7% + 22.4% = 49%. A more conservative rescue rate (60% of viable paths successfully completing) gives ~40%. Either way, the think model's math ability is masked by the truncation ceiling, and the gap with the other models is smaller than the headline numbers suggest.

## Eval Temperature Disaster

All v3b runs use `eval_temperature=0.6` (from MaxRL's eval config). For GPT-OSS and Instruct, this is a mild sharpening that reduces noise. For the think model, it is catastrophic.

The think model's bimodal distribution has a structured mode (good answers) and a raw mode (garbage). At T=0.6, temperature sharpening disproportionately amplifies whichever mode has higher probability. For problems where the raw mode is slightly more probable (many problems, since the base model is not instruction-tuned), T=0.6 pushes all probability mass into the raw mode.

Evidence: truncation rate at eval (T=0.6) is higher than at train (T=1.0). Train truncation runs ~56%; eval truncation reaches 56-70%. The temperature doesn't just sharpen the correct mode. It collapses the bimodal distribution into the wrong peak.

This explains why eval correct (25.3-26.7%) is much lower than train correct (36.4-40.2%). The 10-14pp gap is eval-temperature-induced mode collapse rather than generalization failure.

## Wrong-Boxed Increase: Subreward Optimization

Over 7 steps, wrong-boxed responses tripled (2 to 9 per 32 logged trajectories). The model learned to emit `\boxed{}` more often, improving the format subreward from -0.15 to 0.00, without improving math accuracy. Responses go from "truncated, no answer" (reward = -0.30) to "completed, wrong answer" (reward = 0.00).

This is subreward optimization. The model improves its reward by reducing format penalties rather than solving more problems. With the current reward structure (correct=1.0, format/eos penalties of 0.15 each), the easiest gradient leads toward producing `\boxed{}` with a plausible-looking wrong answer.

## Recommendations

Increase max_tokens to 16,384. This rescues the 40% of truncated responses that were on viable paths. Cost: ~2x longer per-step wall time due to tail latency. Benefit: substantially more correct responses and more gradient signal from mixed-outcome groups.

Eval at T=1.0. The MaxRL eval temperature (0.6) is calibrated for non-thinking models with unimodal distributions. Bimodal think models should be evaluated at train temperature for an accurate picture of capability.

Add explicit format instructions to the system prompt: "Write your final answer as `\boxed{answer}`. For multiple values, use `\boxed{a, b, c}`." The format anxiety accounts for thousands of wasted tokens per response.

Consider dropping think from the experiment matrix unless max_tokens is increased. At 8,192 tokens, the model cannot complete its reasoning process within the budget. No amount of RL training fixes a hard constraint violation.
