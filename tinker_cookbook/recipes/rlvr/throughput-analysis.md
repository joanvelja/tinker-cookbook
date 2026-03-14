# RLVR Throughput Analysis

## Bottleneck Profile

From smoke tests (Qwen3-8B, B=8, G=8, max_tokens=8192):

| Phase | Time | % of total |
|-------|------|-----------|
| Sampling (GPU decode) | ~317s | 97% |
| Training (forward_backward + optim_step) | ~10s | 3% |
| Grading (LLM judge, concurrent) | ~1s | embedded in sampling |

Sampling dominates. All optimizations must attack this.

## Parametric Model

**Invariants** (model/task-determined, not controllable):
- `T_avg` = average tokens per completion (thinking mode determines this)
- `s` = Tinker decode throughput (tok/s, model-size-determined)
- `r` = grader latency per call
- `α` = training cost per token (GPU forward_backward)

**Knobs** (controllable):
- `B` = batch_size (groups per batch)
- `G` = group_size (envs per group, same prompt)
- `N` = n_batches (total training iterations)
- `K` = num_substeps (gradient steps per sampling round)

### Per-batch wall time

```
t_batch = t_sample + t_train

t_sample ≈ T_tail(B·G) / s     # wait for slowest of B·G parallel completions
t_train  = K · α · (B·G/K) · T  = α · B · G · T   # K substeps, each on B·G/K problems
```

For thinking models, most completions hit max_tokens:
```
T_tail ≈ max_tokens  →  t_sample ≈ max_tokens / s  (constant in B, G)
```

### Total training time

Holding total gradient signal constant (`N · B · G · K = const`):

```
t_total = N · (max_tokens/s + α·B·G·T)
```

Increasing B·G (more samples per batch, fewer batches): t_sample/batch is constant,
t_train/batch grows, but N shrinks proportionally. Net effect: **scaling B·G is
approximately free** since t_train << t_sample.

## Optimization Catalog

### Tier 1: Attacks the 97% bottleneck

**`num_samples=G` batched rollout** — Instead of G separate API calls per group,
make 1 call with `num_samples=G`. The Tinker server shares prompt KV-cache and
batches decode across all G sequences in one GPU forward pass.

Why it's fast: single-sequence LLM decode is memory-bandwidth bound (~20% GPU
utilization). Batched decode (G=8) does G× more compute per memory-read cycle,
reaching ~70% utilization. Wall time for G=8 batched ≈ 2-3× a single sequence,
not 8×. Net throughput improvement: **~2-4× on sampling wall time**.

### Tier 2: More learning per sample-dollar

**`num_substeps=K`** — K gradient steps per sampling round, each on a different
slice of the batch. Crucially, each substep trains at UPDATED weights from the
previous substep:

```
weights₀ → grad(chunk₁, w₀) → weights₁ → grad(chunk₂, w₁) → weights₂
```

Multiple SGD steps at updated weights converge faster than one large batch step
(standard optimization theory). Cost: K extra clock cycles (~10s each). Benefit:
>1× learning from the same expensive sampled data. At 97/3 sampling/training
cost split, spending 6% (K=2) instead of 3% on training for meaningfully more
learning is obviously worth it.

### Tier 3: Pipeline optimizations (single-digit %)

**`StreamMinibatchConfig`** — Start training on early-finishing groups while slow
ones are still sampling. Eliminates dead clock cycles spent waiting for the tail.
For thinking models with high variance in completion time (some finish at 2000
tokens, some hit 8192), this is significant — the tail can be 3-5× slower than
the median.

**`AsyncConfig(max_steps_off_policy=K)`** — Fully decoupled sampling and training.
Training loop grabs completed trajectories as they arrive — never waits for the
slowest completion. For thinking models this eliminates tail-latency waste
entirely. Cost: trajectories are slightly off-policy (sampled K steps ago).
Monitor `kl_sample_train_v2` — stable training has KL < 0.01.

Both exist because Tinker charges per token but schedules per clock cycle.
Dead clock cycles (waiting for the tail, waiting between sampling and training)
are wasted money.

**`remove_constant_reward_groups=True`** — Skip groups where all G rollouts got
the same reward (zero gradient signal). From smoke tests, ~25-50% of groups are
constant. Saves that fraction of t_train (~1-2% of total).

**Clock cycle pipelining** — Already automatic in `train_step()`: `forward_backward_async`
and `optim_step_async` are submitted together so they land on the same clock cycle
(~10s). Naive sequential submission wastes 2 extra cycles.

## Derived Invariants from Smoke Tests

| Invariant | Config A (Qwen3 think) | Config B (Qwen3 no-think) | Config C (GPT-OSS) |
|-----------|----------------------|--------------------------|-------------------|
| T_avg | 6418 tok | 1215 tok | 3833 tok |
| T_tail/s | 317s | 147s | 188s |
| α (s/tok) | 2.3×10⁻⁵ | 4.6×10⁻⁵ | 2.6×10⁻⁵ |
| t_train/t_batch | 3% | 2.4% | 3.3% |

## Projected Run Times (100 batches, B=8, G=8)

### Without `num_samples=G` (current)

| Config | t_total | Notes |
|--------|---------|-------|
| A (think) | 100 × 327s = **9.1h** | Sampling-bound |
| B (no-think) | 100 × 151s = **4.2h** | Sampling-bound |
| C (GPT-OSS) | 100 × 194s = **5.4h** | Sampling-bound |

### With `num_samples=G` (estimated 2-3× sampling speedup)

| Config | t_total (optimistic 3×) | t_total (conservative 2×) |
|--------|------------------------|--------------------------|
| A (think) | 100 × 116s = **3.2h** | 100 × 169s = **4.7h** |
| B (no-think) | 100 × 59s = **1.6h** | 100 × 84s = **2.3h** |
| C (GPT-OSS) | 100 × 73s = **2.0h** | 100 × 104s = **2.9h** |

### With `num_samples=G` + larger B·G (same total samples, fewer batches)

Since t_sample is constant in B·G (all samples parallel), doubling B·G halves N
for the same total training signal. Example: B=16, G=8 (128 samples/batch) with
N=50 ≈ same wall time as B=8, G=8 (64 samples/batch) with N=100, but 2× the
per-step gradient quality (larger effective batch).
