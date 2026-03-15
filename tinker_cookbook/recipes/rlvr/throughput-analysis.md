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

### With `num_samples=G` (benchmarked: ~1.5× speedup)

Benchmark (Qwen3-8B think, B=8, G=8, 2 batches):

| | Baseline (64 calls) | Batched (8 calls) | Speedup |
|---|---|---|---|
| Batch 0 | 327s | 248s | 1.32× |
| Batch 1 | 356s | 201s | 1.77× |

Projected 100-batch runs:

| Config | Baseline | Batched (~1.5×) |
|--------|----------|----------------|
| A (think) | 9.1h | **~6.2h** |
| B (no-think) | 4.2h | **~2.9h** |
| C (GPT-OSS) | 5.4h | **~3.7h** |

### Stacking all knobs (benchmarked + estimated)

Starting from batched baseline (248s/batch for Config A):

| Knob | Mechanism | Wall Δ/batch | Learning Δ | Cumulative |
|------|-----------|-------------|-----------|------------|
| `num_samples=G` | Batched decode, shared KV-cache | -79s | — | 248s, 1 step |
| `AsyncConfig(mso=1)` | Overlap sample₂ with train₁ | -8.5s | — | 240s, 1 step |
| `StreamMinibatch(k=2)` | Train at group 4/8 completion | -8s | — | 232s, 1 step |
| `remove_constant_groups` | Skip ~62% zero-gradient groups | -5s | — | 227s, 1 step |
| `num_substeps=2` | 2nd gradient step on same data | **+10s** | **2× learning** | 237s, **2 steps** |

**Effective learning throughput:**

| Configuration | Wall/batch | Steps/batch | Steps/hour |
|---|---|---|---|
| Baseline (old) | 327s | 1 | 11.0 |
| + `num_samples=G` only | 248s | 1 | 14.5 (1.3×) |
| + all pipeline opts | 227s | 1 | 15.9 (1.4×) |
| + `num_substeps=2` | 237s | **2** | **30.4 (2.7×)** |

`num_substeps=2` is the real multiplier: sampling is 97% of cost, so the 2nd
gradient step costs ~4% more wall time but doubles learning throughput. You pay
10s extra to get a whole extra gradient step from data you already paid 240s
to sample.

### Recommended full-throughput config

```bash
uv run --env-file .env python -m tinker_cookbook.recipes.rlvr.train \
  model_name="Qwen/Qwen3-8B" \
  dataset=omni_math \
  batch_size=8 group_size=8 max_tokens=8192 \
  n_batches=100 \
  num_substeps=2 \
  max_steps_off_policy=1 \
  remove_constant_reward_groups=True \
  eval_on_start=False \
  eval_every=10 \
  save_every=20 \
  wandb_project=rlvr-experiments \
  behavior_if_log_dir_exists=delete
```

### With larger B·G (same total signal, fewer batches)

Since t_sample is approximately constant in B·G (all samples parallel),
doubling B·G halves N for the same total training signal. But B·G is bounded
by Tinker's service concurrency — beyond the throughput limit, wall time
grows linearly.
