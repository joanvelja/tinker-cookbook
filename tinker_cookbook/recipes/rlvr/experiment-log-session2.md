# RLVR Experiment Log — Session 2 (2026-03-14 evening → 2026-03-15)

## Active Runs

Three v3b runs on OmniMath-2 with MaxRL-aligned params:
- GPT-OSS-20B (19 steps, learning)
- Qwen3-30B-A3B-Instruct-2507 (16 steps, flat)
- Qwen3-30B-A3B Think (7 steps, flat)

All share: PPO clip 0.2, LR=1e-5, KL=0, grad_clip=0.3, B=128, G=16,
maxrl advantage, format_coef=0.15, eos_coef=0.15, async+streaming.

## Root Finding: Entropy Determines RL Learnability

All three models see the same problems (seed=42, identical dataset split and
batch ordering). GPT-OSS improves while Qwen doesn't. The difference is entropy.

| Model | Entropy (nats) | Perplexity | After 19 steps |
|-------|---------------|------------|----------------|
| GPT-OSS-20B | 0.84 | 2.3 | correct +4pp, boxed +26pp, tok -22% |
| Qwen3 Instruct | 0.33 | 1.4 | flat on all axes |
| Qwen3 Think | 0.27 | 1.3 | flat on all axes |

The Qwen models are near-deterministic at temp=1.0. The policy gradient has
nowhere to push — the token distributions are already sharp. GPT-OSS has 3×
more entropy headroom, so the same gradient signal produces actual behavior
change. This is the binding constraint, not LR or reward design.

## GPT-OSS v3b: What It Learned

Eval at step 10: 54.8% correct on 664 held-out problems.

Reward improvement decomposition (from reward signal probe):
- 86% from penalty reduction (format 55%→82%, fewer truncations)
- 14% from more correct answers (train correct 39.6%→40.9%)

Qualitative decomposition of compression (from reasoning quality probe):
- 60%: format compliance — stop losing to truncation
- 25%: reduced metacognitive overhead — fewer "But/Wait/Actually" spirals
- 15%: genuine reasoning efficiency — faster commitment to correct approach

Self-correction markers dropped 35% in absolute count and 20% in density
(markers per 1000 chars). The model is cutting dead-end exploration and
low-value throat-clearing, not cutting useful reasoning.

Step 0 pathology: "119 is a multiple of which number?" → 3307 chars, 18 "But"
markers of metacognitive spiraling on a trivial factoring question.

Step 15 improvement: "Find four numbers whose products sum to primes" → 2641
chars, 5 markers. Tries one example (1,2,3,5), works, stops.

Analysis channel median length: 19,635 → 9,169 chars (-53%). The model still
uses the analysis channel 100% of the time — it learned to think shorter, not
to stop thinking. Analysis-channel-only death mode (never reaches final
channel): 10/32 → 6/32, declining.

## Qwen Instruct v3b: Template Lock

The SFT style prior (`### Step N:` markdown headers) is fully in control. RL
hasn't dented it after 16 steps. Step 0 and step 15 responses look structurally
identical — same template, different nouns.

Failure mode taxonomy (32 logged trajectories per step):

| Category | Step 0 | Step 5 | Step 10 | Step 15 |
|----------|--------|--------|---------|---------|
| Correct + concise (<5k chars) | 5 | 7 | 6 | 5 |
| Correct + verbose (>5k chars) | 9 | 9 | 10 | 11 |
| Wrong + boxed (arithmetic) | 0 | 1 | 1 | 0 |
| Wrong + boxed (wrong approach) | 2 | 1 | 4 | 4 |
| Wrong + boxed (grader FN) | 4 | 4 | 3 | 1 |
| Truncated (saveable) | 7 | 3 | 6 | 6 |
| Truncated (stuck in loop) | 5 | 7 | 2 | 4 |

Improvements and regressions offset: verbose correct up (+2), wrong approach
up (+2), grader FN down (-3). True accuracy (including grader FN) actually
declined slightly: 18/32 → 17/32. The grader is teaching the model to avoid
correct-but-awkward answer formats — a toxic training signal.

Concise correct (category A) unchanged at 5/32. The model is not learning
compression. Token counts flat at 6700 throughout.

## Qwen Think v3b: Post-Solution Tail Bloat

The model finds correct answers early but wastes 80-90% of the think block on
post-answer activity: verification, alternative approaches, hedging, and
format anxiety.

Think block internal structure (from all correct responses):
- Median position of first correct \boxed{}: >95% of think block
- No trend toward earlier answer discovery across 7 steps
- Post-answer rumination is INCREASING (step 6 highest rates)

Post-answer activity breakdown:
- Verification ("let me check once more") — sometimes useful
- Alternative approach ("alternatively, using coordinates") — pure waste
- Hedging ("wait, but maybe I need to reconsider") — almost always confirms
  the original answer
- Format anxiety ("should I write \boxed{(a,b)} or two separate boxes?") —
  one response burned 10,423 chars on this

Truncated responses (44% of all):
- 40% were on viable paths (had found correct answer, ran out of tokens)
- 60% were genuine dead ends (algebraic loops, case enumeration explosions)

Wrong-boxed tripled (2→9 in logged samples). The model emits \boxed{} more
often but with wrong answers. Codex assessment: "subreward optimization, not
healthy stage-1." The model improves reward from -0.30 (truncated) to 0.00
(wrong-boxed) without improving math.

## Grader Issues (Quantified)

False negative rate on wrong-but-boxed answers:
- GPT-OSS v3b: 14% (1/7) — symbolic notation vs English
- Instruct v3b: 24% (5/21) — LaTeX \text{}, simple answers, double-boxing
- Think v3b: found \boxed{answer} literal accepted as correct (grader bug)

Specific failure modes:
- `\text{D}` vs `D` — LaTeX wrapper confuses grader
- `f(n)=n` vs `f(x)=x` — variable name difference
- `1` vs `1` — grader rejects simple correct answers
- Double-boxing: model writes `\boxed{7×2023}` then `\boxed{C}`, extract_boxed
  takes last one → grader gets "C" → incorrect

Grader prompt updated with examples covering these cases (committed, will
affect future runs but not current ones).

## Timing and Throughput

Per-step timing at B=128, G=16 (2048 episodes/step):
- GPT-OSS: ~15-20 min/step (4.5-5.5k tok/resp)
- Instruct: ~20-25 min/step (6.5-6.8k tok/resp)
- Think: ~30-35 min/step (7.3-7.5k tok/resp)

Async+streaming minibatches: first 3 minibatches consume in <1s, last
minibatch blocks for ~200-280s waiting for slowest sampling group. The tail
latency dominates — streaming helps marginally but doesn't solve it.

OpenAI grader rate limiting on instruct v3b: 2048 concurrent grading calls
cause API retries (0.5s backoff). Adds latency but doesn't crash.

## Hypotheses for Next Steps

From Codex + probes converging:

1. Qwen entropy bottleneck: raise temperature to 1.2-1.5 or increase LR to
   2-3e-5. The policy can't reshape at perplexity 1.3.

2. Few-shot exemplar for Qwen: a single concise math example could break the
   markdown-heavy template prior more effectively than 100 RL steps.

3. Think model format instructions: clearer guidance for multi-answer problems
   would eliminate the format-anxiety token drain.

4. GPT-OSS is the one to watch. Next eval at step 20 determines if
   compression translates to held-out accuracy improvement.

5. Increase max_tokens for think model (12k-16k) — 40% of truncated responses
   had found the correct answer but ran out of tokens.

## Killed Runs (This Session)

- GPT-OSS v2b: killed at batch 20. Verbosity spiral from KL=0.03 anchoring
  to base policy verbosity. Eval showed math improving (P(right|boxed) 65%→73%)
  but format collapsing (85%→67% boxed).

- Instruct v2b: killed at batch 15. Old config (IS loss, LR=5e-4, KL=0.03).
  One eval point at 58.6% correct — decent but no second eval to trend.

## Key Architectural Insight

GPT-OSS's Harmony format (analysis → final channel split) is structurally
easier to compress than Qwen's `<think>...</think>`. The analysis/final
transition gives the model a clean "stop thinking, start answering" signal.
Qwen's think block has no equivalent — the model must learn when to close
`</think>`, which requires a harder structural behavior change.

Combined with the 3× entropy gap, this explains why GPT-OSS compresses in 14
steps while Qwen models are flat at 16. The same reward signal, same LR, same
PPO config — different model architecture and entropy profile produce
fundamentally different learning dynamics.
