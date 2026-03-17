# Experimental Setup and Hyperparameters

## Model Matrix

Three model families on OmniMath-2 (4,428 problems, 85/15 train/eval split):

| Model | Size | Renderer | Thinking mode |
|-------|------|----------|---------------|
| GPT-OSS-20B | 20B | `gpt_oss_medium_reasoning` | Harmony analysis channel |
| Qwen3-30B-A3B-Instruct-2507 | 30B (3B active) | `qwen3_instruct` | Never thinks |
| Qwen3-30B-A3B | 30B (3B active) | `qwen3` | Visible `<think>` blocks |

A fourth config (Qwen3-30B-A3B with `qwen3_disable_thinking`) was dropped in v1. The renderer prepends `<think>\n\n</think>` as a suppression hint, but the base model ignores it and generates think blocks anyway.

## Configuration Evolution

### v1 (killed after ~10 batches)

| Parameter | Value | Problem |
|-----------|-------|---------|
| LR | 5e-4 (auto via `get_lr`) | 50x too high for RL |
| Loss | `importance_sampling` | No ratio clipping |
| KL coef | 0.0 | No stabilization |
| Grad clip | 0.0 | Unbounded updates |
| Batch size | 16 | High variance |
| Group size | 4 | Insufficient for MaxRL advantages |
| `num_substeps` | 2 | Accelerated drift on stale data |
| `remove_constant_groups` | True | Selection bias toward lucky groups |

GPT-OSS collapsed within 19 batches. Responses devolved into repeating "medium medium medium," the word from the system prompt `"Reasoning: medium"` becoming a degenerate attractor. KL spiked from 0.008 to 0.074. Root cause: no clipping, no KL penalty, and high LR allow a single lucky batch to produce unbounded policy updates.

### v2 (ran ~30-40 batches per model)

| Change | From | To | Rationale |
|--------|------|----|-----------|
| KL coef | 0.0 | 0.03 | Prevent v1-style collapse |
| Group size | 4 | 8 | More diverse rollouts per prompt |
| Eval fraction | 0.15 (664) | 0.025 (~110) | Eval was taking 2-3h at 8k max_tokens |
| Log path | auto | explicit per run | Prevent collisions on same model_name |

KL penalty at 0.03 produced ~0.0005 effective penalty against rewards of 0.2-0.4. Functionally zero. It prevented catastrophic collapse but did not meaningfully constrain the policy. The verbosity spiral in GPT-OSS v2 (truncation 6% to 33% over 30 steps) was driven by the 10:1 reward asymmetry between correctness (+1.0) and format penalty (-0.1), which KL at this scale cannot counteract.

### v3b (production runs, 75 steps for GPT-OSS)

All three models share identical training infrastructure:

| Parameter | Value | Source |
|-----------|-------|--------|
| LR | 1e-5 | MaxRL 1e-6 full-param, scaled 10x for LoRA |
| Loss | PPO | Clip ratio [0.8, 1.2] (symmetric) |
| KL coef | 0.0 | MaxRL: `use_kl_loss=False`, `use_kl_in_reward=False` |
| Grad clip | 0.3 | MaxRL verbatim |
| Batch size (B) | 128 | MaxRL uses 256; our GPU budget covers 128 |
| Group size (G) | 16 | MaxRL verbatim |
| Advantage | MaxRL | Per-group advantage with alpha=0.5 subgroups |
| LoRA rank | 32 | Standard for 20-30B models |
| Max tokens | 8,192 | 2x MaxRL's 4,096 to accommodate thinking models |
| Train temperature | 1.0 | MaxRL verbatim |
| Eval temperature | 0.6 | MaxRL `val_kwargs.temperature=0.6` |
| Eval top_p | 0.95 | MaxRL `val_kwargs.top_p=0.95` (was not wired in prior runs; now plumbed) |
| Format coef | 0.15 | Per-reward penalty for missing `\boxed{}` |
| EOS coef | 0.15 | Per-reward penalty for truncation (missing EOS) |
| Seed | 42 | Deterministic dataset split and batch ordering |
| Grader | LLM (gpt-5-mini) | `reasoning_effort=medium`, math-aware prompt |
| `remove_constant_groups` | True | Skip zero-gradient groups |
| Async | `max_steps_off_policy=1` | Overlap next sampling with current training |
| Stream minibatches | 4 | Begin training when 25% of groups complete |
| `n_batches` | 500 | ~150 steps = 5 epochs is sufficient |
| Episodes/step | 2,048 | B=128 x G=16 |

## Reward Function

Two-signal reward with parse gating:

```
reward = correct + format_coef * (boxed - 1) + eos_coef * (eos - 1)
```

Variables: `correct` in {0, 1} (LLM grader verdict via gpt-5-mini), `boxed` in {0, 1} (1 if response contains `\boxed{...}`), `eos` in {0, 1} (1 if response terminated with EOS, i.e. was not truncated at `max_tokens`), with `format_coef = 0.15` and `eos_coef = 0.15`.

The `parse_success` gate: if the response is truncated (no EOS), it receives `correct=0` regardless of content. This prevents the model from receiving positive reward for partially-correct truncated responses, which would reinforce verbosity.

Reward range: [-0.30, 1.00]. A correct, well-formatted, non-truncated response scores 1.0. A truncated response with no boxed answer scores -0.30. The 3.3:1 ratio between best and worst outcomes may be too weak to penalize truncation aggressively. v2's 10:1 ratio for correct-vs-format drove the verbosity spiral, but a larger truncation penalty (eos_coef >= 0.3) warrants testing.

## LoRA Learning Rate Derivation

MaxRL uses LR=1e-6 for full-parameter Qwen3-1.7B training. We use LoRA (rank 32) instead, following the 10x rule from "LoRA Without Regret" (Thinking Machines Lab): LoRA gradients are projected into a rank-r subspace, concentrating gradient energy into fewer parameters. The effective per-parameter learning rate is ~10x lower than the headline LR, so LoRA needs 10x higher LR to match full-parameter training dynamics.

1e-6 (full-param) x 10 (LoRA correction) = **1e-5**.

For comparison, Tinker's `get_lr()` returns ~5e-4 for these models. That default is calibrated for supervised learning, where the loss surface is smoother and higher LR is tolerable. The 50x gap between SL-calibrated and RL-appropriate LR explains the v1/v2 instability.

## Comparison with Published Methods

| Parameter | Ours (v3b) | MaxRL | DAPO | DeepSeek-R1 |
|-----------|-----------|-------|------|-------------|
| Batch size (prompts) | 128 | 256 | 512 | 32 |
| Group size | 16 | 16 | 16 | 16 |
| LR | 1e-5 (LoRA) | 1e-6 (full) | | 3e-6 |
| KL | 0.0 | 0.0 | 0.0 | 0.001 |
| Clip ratio | 0.2/0.2 | 0.2/0.2 | 0.2/0.28 | 10.0 |
| Grad clip | 0.3 | 0.3 | | |
| PPO epochs | 1 | 1 | | |
| Max response tokens | 8,192 | 4,096 | | 32,768 |
| Total epochs | ~2 (75 steps) | 5 | | |
| Advantage | MaxRL | MaxRL | GRPO variant | GRPO |
| Model params trained | LoRA rank-32 | Full | Full | Full |

Key differences from MaxRL: LoRA instead of full fine-tuning; 2x longer max response length to accommodate thinking models; smaller batch size (128 vs 256) due to compute budget; added format and EOS reward signals (MaxRL uses correctness only).

## Throughput

Per-step timing at B=128, G=16 (2,048 episodes/step):

| Model | Wall time/step | Avg tokens/response |
|-------|---------------|---------------------|
| GPT-OSS | 15-20 min | 4.5-5.5k |
| Qwen3 Instruct | 20-25 min | 6.5-6.8k |
| Qwen3 Think | 30-35 min | 7.3-7.5k |

Sampling accounts for 97% of wall time. Training (forward_backward + optim_step) takes ~10s per step. The async + streaming configuration overlaps sampling with training: the first 3 of 4 minibatches are consumed in <1s, with the last minibatch blocking for ~200-280s on the tail of the sampling distribution.

The `num_samples=G` optimization (batched decode with shared KV-cache) provides 1.3-1.8x speedup on sampling. Combined with all pipeline optimizations, effective throughput reaches ~30 gradient steps/hour (2.7x over naive baseline).
