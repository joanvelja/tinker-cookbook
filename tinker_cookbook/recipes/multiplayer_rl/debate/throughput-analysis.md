# Debate Training: Parametric Wall-Clock Model

## Episode Structure

SEQUENTIAL protocol, R rounds. Per episode:

```
[A₀] → [B₀] → [A₁] → [B₁] → ... → [judge]
 t_s     t_s    t_s     t_s           t_j
```

- **2R** debater slots (serial within episode)
- **1** judge call at the end
- Self-play: both A,B use trained policy (2 trajectories per episode)
- Frozen-opp: trained agent + opponent model (1 trajectory per episode)

SIMULTANEOUS protocol cuts the critical path:
```
[max(A₀,B₀)] → [max(A₁,B₁)] → ... → [judge]
```
R slots instead of 2R. **~50% savings on debater time.**

## Parallelism

- **Across problems (B groups):** fully parallel via `asyncio.gather`
- **Across games within group (G episodes):** fully parallel via `asyncio.gather`
- **Within an episode:** strictly serial (turns depend on previous turns)
- **Constraint:** Tinker service concurrency cap `C = sampling_max_connections`

## The Formula

```
T_batch = max(t_episode, B·G·calls_per_ep / C) + T_train + T_ckpt
          ↑ critical path   ↑ throughput limit     ↑ GPU     ↑ fixed
```

Where:
```
t_episode = 2R·t_s + t_j                    (SEQUENTIAL)
t_episode = R·t_s + t_j                     (SIMULTANEOUS)
calls_per_ep = 2R + 1                       (SEQUENTIAL self-play)
calls_per_ep = R + 1                        (SIMULTANEOUS self-play)
```

Two regimes:
1. **Critical-path limited** (`t_episode > B·G·calls/C`): adding more B·G is free
2. **Throughput limited** (`B·G·calls/C > t_episode`): wall time grows linearly with B·G

## Scaling Analysis

### With B (batch_size)

| Regime | Effect of 2×B | Gradient quality |
|--------|--------------|-----------------|
| Critical-path | **Free** (t_episode unchanged) | sqrt(2)× better |
| Throughput | **2× slower** per batch | sqrt(2)× better |

Gradient signal per wall-second: `sqrt(B) / T_batch(B)`. Maximized at the
crossover point between regimes.

### With G (group_size)

Same scaling as B for wall time. But gradient quality benefit saturates:
- G=2: noisy advantages
- G=4: reasonable
- G=8: good
- G=16+: diminishing returns

**Recommendation: G=4, maximize B up to throughput limit.**

### With R (num_rounds)

- t_episode linear in R
- calls_per_episode linear in R
- Both regimes get linearly slower
- Debate quality may saturate at R=2-3

**R=2 is likely near-optimal for quality/cost.**

### With protocol_kind

SIMULTANEOUS vs SEQUENTIAL: **~50% savings** on the critical path term
(R slots vs 2R). The throughput term also halves (R+1 vs 2R+1 calls).

## Sweet Spots

For self-play SEQUENTIAL R=2 on a mid-size model (t_s ≈ 10s, t_j ≈ 10s):

```
t_episode = 4·10 + 10 = 50s
calls_per_ep = 5
```

| B | G | Total games | T_rollout (C=256) | Regime |
|---|---|------------|-------------------|--------|
| 8 | 4 | 32 | max(50, 32·5·10/256) = 50s | Critical-path |
| 16 | 4 | 64 | max(50, 64·5·10/256) = 50s | Critical-path |
| 32 | 4 | 128 | max(50, 128·5·10/256) = 50s | Critical-path |
| 32 | 8 | 256 | max(50, 256·5·10/256) = 50s | Crossover |
| 64 | 8 | 512 | max(50, 512·5·10/256) = 100s | Throughput |

**The sweet spot is where B·G·calls/C ≈ t_episode.** Below that, adding
more B·G is free. Above that, you're paying linearly.

For C=256, R=2, t_s=10s: sweet spot at `B·G ≈ C/(calls·t_s/t_ep) = 256·50/(5·10) = 256`.

So **B=32, G=8** or **B=64, G=4** with C=256 is approximately optimal.

## Available Optimizations

| Optimization | Effect | Status |
|---|---|---|
| SIMULTANEOUS protocol | ~50% critical path | Code exists |
| `num_minibatches=2-4` (streaming) | Overlaps sampling + training | Already wired in CLI |
| `num_substeps=2` | Extra gradient steps per sampling round | Available, not exposed |
| Reduce `max_tokens` | Reduces t_s | Config knob |
| Reduce `judge_max_tokens` | Reduces t_j (on critical path!) | Config knob |
| `AsyncConfig` | Eliminate tail-latency waste | Available, not wired |
| Raise `sampling_max_connections` | Push throughput limit higher | Config knob |

**The #1 lever for debate is SIMULTANEOUS protocol** — it halves the dominant term.
For RLVR it was `num_samples=G`; for debate it's parallelizing debater turns.

## Stacking Knobs (Debate, SEQ self-play R=2, B=32 G=4, t_s=10s)

Baseline: `t_episode = 50s`, throughput-limited at `B·G=128 → T_batch ≈ 53s`.

| Knob | Mechanism | Wall Δ/batch | Learning Δ |
|------|-----------|-------------|-----------|
| Baseline | — | 53s | 1 step |
| `num_minibatches=2` | Overlap training with tail | -2s | — |
| `num_substeps=2` | 2nd gradient step | +10s | **2× learning** |
| `remove_constant_groups` | Skip zero-gradient groups | -1s | — |
| SIMULTANEOUS protocol | R slots instead of 2R | **-20s** | — |

| Configuration | Wall/batch | Steps/batch | Steps/hour |
|---|---|---|---|
| SEQ baseline | 53s | 1 | 67.9 |
| + streaming + remove_const | 50s | 1 | 72.0 (1.06×) |
| + `num_substeps=2` | 60s | **2** | **120.0 (1.77×)** |
| Switch to SIM protocol | 33s | 1 | **109.1 (1.61×)** |
| SIM + `num_substeps=2` | 43s | **2** | **167.4 (2.47×)** |

**Key insight**: for debate, SIMULTANEOUS protocol and `num_substeps=2` are
multiplicative. SIM halves the critical path; substeps double learning per
sampling round. Combined: **~2.5× effective learning throughput vs SEQ baseline**.

## RLVR vs Debate: Optimization Comparison

| | RLVR | Debate |
|---|---|---|
| **#1 lever** | `num_samples=G` (batched sampling) | SIMULTANEOUS protocol |
| **Why** | Same prompt G times → share KV-cache | Debaters can act in parallel |
| **Measured speedup** | 1.5× wall time | ~2× wall time (estimated) |
| **`num_substeps`** | 2.7× learning/h | 2.5× learning/h |
| **Async/streaming** | 3-8% wall time | 3-6% wall time |
| **`num_samples=G`** | **Yes** (single-turn, shared prompt) | **No** (multi-turn, different prompts) |

## Recommended Configs

### Debate — Maximum throughput (SEQ self-play)
```
batch_size=32 group_size=4 num_rounds=2
num_minibatches=2 num_substeps=2
```

### Debate — Maximum throughput (SIM self-play)
```
batch_size=64 group_size=4 num_rounds=2
num_minibatches=2 num_substeps=2
```
(Higher B because SIM has fewer calls per episode → higher throughput limit.)
