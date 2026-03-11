# RL on Tinker: Findings, Bottlenecks, and Practical Levers

## Purpose

This document synthesizes what we learned from diagnosing slow debate-style RL runs in this repository, and turns those findings into an execution guide for throughput-focused RL work on Tinker.

Motivating baseline (single local run; add date/commit/log artifact when using this as a benchmark reference):

- Runtime: about 21 minutes
- Workload: 2 problems x `group_size=2` = 4 rollouts
- Debate setup: 2 rounds, `SEQUENTIAL`
- Calls per rollout in this setup: 4 debater turns + 1 final judge call = 5
- Total calls: 4 x 5 = 20
- Model fleet: debaters on `Qwen/Qwen3-235B-A22B-Instruct-2507`, judge on `Qwen/Qwen3-30B-A3B-Instruct-2507`

Back-of-envelope only (not equivalent to per-actor latency): 21 minutes / 20 calls is about 63 seconds per call averaged over mixed actors/models and partial overlap. Per-actor latency must be measured directly.

---

## Scope and ceteris-paribus framing

In this report, ceteris paribus means:

- Keep fixed: models, token caps, protocol/rounds, problem set, `group_size`, sampling params, seed policy, retry config, SDK/cookbook versions.
- Vary only: execution topology and scheduling behavior.

Why this matters:

- If you change model/token/protocol while changing scheduling, attribution is no longer clean.

---

## Quick Glossary

- `builder`: one problem-level object returned by dataset batching that creates envs for that problem.
- `env group`: the `group_size` environments under a builder.
- `critical path`: serial dependencies that determine minimum wall time.
- `holder/session`: SDK-level shared state behind `ServiceClient`/`SamplingClient` that includes dispatch/backoff behavior.
- `arm`: one experimental variant (for example `A0`, `A1`).
- `replicate`: one repeated run of an arm under the same controls.
- `block`: one time-local set containing exactly one run from each arm, used to control temporal service drift.

---

## System Model in Plain Language

A rollout is not a single inference. It is a protocol-driven sequence of inference calls coordinated by runtime and env code.

For the specific smoke setup discussed here:

- `ProtocolKind.SEQUENTIAL`
- `num_rounds=2`
- `include_judge_turns=False`
- default `LLMJudgeCallback` where `on_boundary` is a no-op and `on_final` performs one judge sample

Per rollout critical path in this exact setup:

1. Debater A propose
2. Debater B propose
3. Debater A critique
4. Debater B critique
5. Judge final verdict sample

Caveat: the 5-call count is setup-specific; other protocol/settings change it.

Repo anchors:

- [smoke_judge_exploit.py](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/scripts/smoke_judge_exploit.py)
- [schedule.py](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/core/schedule.py)
- [env.py](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/env.py)
- [rollouts.py](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/rl/rollouts.py)
- [runtime.py](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/core/runtime.py)

---

## Evidence Taxonomy

- Code-backed: directly verifiable from repository and local SDK code.
- Probe-backed: ad-hoc local measurements; directional unless artifacted with script + raw logs + commit SHA.
- External-signal: issue tracker/docs evidence of plausible confounds; not direct attribution for one run.

---

## Operator Playbook (Immediate Next Moves)

If your objective is throughput under fixed model/token/protocol settings:

1. Implement only `A1` first (problem-level parallelization), keep everything else fixed.
2. Run blocked interleaving of `A0` and `A1` for an initial decision pass.
3. Collect per-call telemetry (`wall_s`, retries, 429 flags, backoff seconds, token counts).
4. Promote only if median run wall time improves and p95 does not regress materially.
5. Then test topology arms (`A2/A3/A4/A5`) to isolate holder/session coupling effects.

---

## Condensed Synthesis

### Established (Code-backed, High Confidence)

1. There is real problem-level serialization in the current smoke topology.
2. Within-rollout seriality is semantic in this protocol, not a bug.
3. Shared clients do not imply deterministic one-at-a-time execution by default.
4. Holder/session-level backoff and dispatch coupling is real and can cross-couple actors.
5. Retry defaults can create long apparent hangs.
6. Runtime awaits callbacks under lock by design; in this smoke path boundary callback cost is effectively zero.
7. A dead/intermediate `next_ob` build exists and is likely low impact relative to model/service wait time.

### Directional (Probe-backed, Medium Confidence)

1. Synthetic topology mirrors suggest meaningful upside from eliminating problem-level serialization.
2. Small-model exploratory probes did not show deterministic shared-client serialization, but did show tail jitter.
3. Stream/async RL overlap knobs may provide incremental gains; effect size for this workload remains unmeasured here.

### External-signal (Plausible Confounds, Not Direct Attribution)

Open issues indicate active service/SDK tail-risk classes:

- sampling stalls and retry exhaustion reports
- `save_weights_and_get_sampling_client` hanging report
- seed+`num_samples>1` duplication gotcha
- batched multi-prompt sampling request

These establish plausibility of non-local contributors but do not independently prove root cause for your run.

---

## Findings in Detail

### [Code-backed | High confidence] 1) Problem-level scheduling is the strongest local bottleneck candidate

Current smoke flow processes builders in a sequential outer loop while only gathering envs within each builder.

- [smoke_judge_exploit.py:149](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/scripts/smoke_judge_exploit.py:149)
- [smoke_judge_exploit.py:150](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/scripts/smoke_judge_exploit.py:150)
- [smoke_judge_exploit.py:158](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/scripts/smoke_judge_exploit.py:158)

Implication:

- With 2 problems and `group_size=2`, there are 4 rollout units total, but only one builder active at a time.
- Effective concurrency is narrower than workload cardinality.

### [Code-backed | High confidence] 2) Per-rollout seriality is expected protocol behavior

`do_single_rollout` alternates policy sample and env step with explicit awaits.

- [rollouts.py:31](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/rl/rollouts.py:31)
- [rollouts.py:32](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/rl/rollouts.py:32)

This defines baseline critical-path structure and should not be misdiagnosed as accidental Python serialization.

### [Code-backed | High confidence] 3) Shared client does not automatically mean hard serialization

From local SDK `tinker==0.13.1` internals:

- request-per-client cap constant exists at 50
- client pools reuse then create additional internal clients as needed
- dispatch semaphores are used for sample flow control

These mechanisms mean shared client usage is not equivalent to strict single-flight semantics by default.

### [Code-backed | High confidence] 4) Holder/session topology can create cross-role interference

Also in local SDK internals:

- holder-level backoff state exists (`_sample_backoff_until`)
- dispatch limits are holder-scoped
- 429 handling writes backoff state used by sampling path

Practical consequence:

- opponent and judge sharing holder/session can couple tail behavior under load.

Experiment design implication:

- separate `SamplingClient`s on one `ServiceClient` (`split_sampling`) is not the same treatment as separate `ServiceClient`s (`split_service`).

### [Code-backed | High confidence] 5) Retry defaults can produce long apparent hangs

Local SDK retry path supports long no-progress windows and exponential backoff.

- `RetryConfig.progress_timeout = 120 * 60`
- retry behavior with jitter/backoff
- `create_sampling_client(..., retry_config=...)` allows overrides

Implication:

- some long stalls can be retry-policy tails even with correct local logic.

### [Code-backed | High confidence] 6) Runtime lock and judge callback behavior must be contextualized

Runtime awaits callbacks under lock by design for semantic correctness.

- [runtime.py:109](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/core/runtime.py:109)
- [runtime.py:216](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/core/runtime.py:216)
- [runtime.py:223](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/core/runtime.py:223)

For this smoke path specifically:

- boundary callback returns `None`
- final judge sample remains one critical-path call per rollout

So "judge off critical path" should be treated as unproven gain until timed decomposition confirms benefit.

### [Code-backed | High confidence] 7) Dead/intermediate observation build is real but probably low leverage

`DebateEnv.step` builds `next_ob` before branch logic and can rebuild later.

- [env.py:123](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/env.py:123)

This is worth cleanup for clarity, but unlikely to move wall time materially compared with sampling waits.

### [Probe-backed | Medium confidence] 8) Topology probe signals are directional unless artifacted

Current probe conclusions should be treated as hypothesis-generating unless benchmark artifacts are attached (script path, commit SHA, raw timing logs).

Recommended conservative wording:

1. Synthetic topology mirror suggests potential upside from problem-level parallelization.
2. Small-model probe suggests no deterministic shared-client hard serialization, with visible tail jitter.
3. Impact of stream/async RL overlap for this exact workload is unmeasured in this document.

### [Code-backed + External-signal | Medium confidence] 9) Seed caveat is throughput-adjacent but quality-critical

- seed with `num_samples > 1` can increase duplicates (issue reports)
- scripts also warn seeded behavior can remain nondeterministic across separate calls

For this exact smoke path (`num_samples=1`), this is not the dominant throughput lever, but it matters for RL/eval validity.

### [Code-backed | Medium confidence] 10) KV/extension statements should remain cautious

Safe statement:

- dynamic debate prompts are generally unfriendly to easy prefix-sharing interpretations.
- no explicit user-facing delta-continuation lever is exposed in the path used here.

Overstatement to avoid:

- "KV reuse is impossible".

Correct phrasing:

- KV/extension effects may exist internally, but are not currently a clean, directly controlled lever in this recipe and were not measured in this investigation.

---

## Symptom Triage Map

| Symptom | Likely layer | Check first | Next action |
|---|---|---|---|
| Long silent waits with low stdout activity | Retry/backoff/service tails | retry count, 429 rate, `backoff_s`, progress timeout | tighten telemetry, compare runs with explicit retry config |
| Judge lag only when sharing topology | Holder/session coupling | compare `split_sampling` vs `split_service` | split by `ServiceClient` and re-measure |
| Mean improves but p95/p99 worsens | Tail amplification | run-level p95/p99 deltas by block | reject or gate rollout on tail criteria |
| Little gain despite async usage | Topology still narrow | concurrency-over-time + dynamic ceiling | widen outer scheduling (`builder` level) |
| Inconsistent speedup across day/time | service drift confound | per-block deltas, blocked randomization | increase block count; paired analysis |

---

## Official Docs: Operational Takeaways

1. Sampling API is single prompt + `num_samples` and standard sampling params.
- https://tinker-docs.thinkingmachines.ai/api-reference/samplingclient
- https://tinker-docs.thinkingmachines.ai/api-reference/types

2. Async overlap/pipelining is central to throughput.
- https://tinker-docs.thinkingmachines.ai/async
- https://tinker-docs.thinkingmachines.ai/under-the-hood

3. Async/streaming RL overlap features are documented as experimental.
- https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams

4. OpenAI-compatible endpoint is beta/testing-oriented rather than primary high-throughput path.
- https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

---

## General Levers for RL on Tinker

### A) Model the concurrency graph explicitly

Track parallel width separately at problem, rollout, stage, and callback layers.

### B) Separate coupling layers before optimizing

Disentangle:

- protocol-imposed seriality
- Python scheduling topology
- runtime lock behavior
- holder/session coupling
- service queue/retry tails

### C) Use tail metrics as first-class outcomes

Report p50/p95/p99 and retry/backoff incidence by actor/phase, not means only.

### D) Lock reproducibility and provenance

Every claim should include:

- run manifest (models/params/tokens)
- SDK/cookbook SHA/version
- retry config
- run timestamp window and block id

### E) Maintain observability hygiene

Use unbuffered output plus structured event telemetry so "hang vs slow" is distinguishable.

---

## Controlled Experiment Design (Ceteris Paribus)

### Fixed controls for all arms

1. models, token caps, protocol/rounds, problem set, `group_size`
2. sampling params (`temperature`, `top_p`, `top_k`, `num_samples`, stop sequences)
3. seed policy
4. retry/backoff config
5. SDK/cookbook revision
6. warmup policy and run-window rules

If a control cannot be fixed, record it as a covariate.

### Factorized arm matrix

Define factors:

- `F1` scheduling: problem-serial vs problem-parallel
- `F2` topology: shared vs `split_sampling` vs `split_service`

Arms:

- `A0`: serial + shared (baseline)
- `A1`: parallel + shared
- `A2`: serial + `split_sampling`
- `A3`: serial + `split_service`
- `A4`: parallel + `split_sampling`
- `A5`: parallel + `split_service`

This avoids collapsing distinct topology effects and preserves interaction visibility.

### Blocked execution protocol

1. Use blocked randomization: each block contains exactly one run of every arm (`A0..A5`) in randomized order.
2. Keep problem list and order identical within each block across arms.
3. Rotate problem order between blocks to avoid deterministic order artifacts.
4. Run warmup before measurement and exclude warmup from analysis.
5. Primary endpoint: run-level total wall time.
6. Secondary endpoints: actor/phase call latency, retries, 429 rate, backoff seconds.

### Sample size and uncertainty guidance

- For stable run-level median/p95 comparisons, use at least about 20 blocks.
- For call-level p99 claims, target roughly >=1000 calls per arm.
- Use paired bootstrap over blocks for run-level deltas/ratios.
- Use hierarchical bootstrap (`block -> run -> call`) for call-level summaries.

### Decision rule template

Treat changes as accepted only when all conditions hold:

1. run-level median wall time improves meaningfully vs `A0`
2. run-level p95 does not regress materially
3. retry/backoff incidence does not worsen beyond pre-set tolerance

Record thresholds before running to avoid post-hoc cherry-picking.

---

## Visualization Pack (Designed to Pop and Diagnose Fast)

### 1) Timeline chart (Gantt: each call as a time bar)

- x-axis: wall-clock
- y-axis: lane = `block/arm/problem/env/actor/phase`
- visual encoding: color by actor, pattern for retries/backoff

Use segmented phases if available (`submit->first_token`, generation, retry/backoff, callback).

### 2) Concurrency-over-time chart

- step plot of in-flight calls over time
- overlay dynamic (phase-conditioned) concurrency ceiling, not one flat ceiling line

### 3) Latency distributions

- run-level median points with CIs by arm/actor/phase
- pooled call-level distribution plots for diagnostics only

Do not treat pooled call points as IID for inference.

### 4) Critical-path decomposition

- waterfall by arm using run-level median component times
- components: sampling, retry/backoff wait, callback/runtime overhead, other Python overhead

### 5) Tail-risk curve (ECDF: fraction finished under latency threshold)

- include full curve plus p95-p100 zoom
- log-scale x-axis can help reveal tail separation

### 6) Token-length effects

Instead of simple `input_tokens` vs latency scatter inference, model partial effects:

- `latency ~ input_tokens + output_tokens + actor + phase + retries + arm`

Use scatter as a diagnostic companion plot only.

### 7) Speedup summary

- forest plot of paired arm-vs-`A0` ratios by block
- pooled effect with 95% CI

This makes temporal drift visible instead of hidden.

### Minimal event schema required

One row per call with at least:

- run metadata: `run_id`, `block_id`, `arm`, `replicate`
- identity: `problem_id`, `env_id`, `call_id`, `attempt_id`
- context: `actor`, `phase`
- timing: `t_submit`, `t_first_token` (if available), `t_last_token`, `t_callback_done`, `wall_s`
- volume: `input_tokens`, `output_tokens`, `max_tokens`
- reliability: `retries`, `had_429`, `backoff_s`, `status`, `error_code`
- topology trace: `service_client_id`, `sampling_client_id`

Without this schema, critical-path and tail attribution are hard to audit.

---

## Caveats and Non-Findings

1. Do not claim exact speedup magnitudes without artifacted benchmarks.
2. External issues are plausibility evidence, not direct root-cause proof for your run.
3. Avoid absolute KV-cache claims; current evidence supports "not a clean controlled lever here," not "impossible."
4. Stream/async overlap features may help but should be treated as experimental until validated on your workload.

---

## Bottom Line

The right conclusion is mixed-system, not single-cause:

- There is a real local scheduling bottleneck (problem-level serialization) under fixed model/token settings.
- There are credible service/SDK tail contributors (retry/backoff/429 dynamics) that can dominate some runs.

So the correct operational strategy is:

1. run ceteris-paribus topology ablations,
2. instrument per-call timing and retry signals,
3. analyze with blocked paired statistics,
4. decide from run-level effects plus tail-risk guardrails.

That is the shortest path to attribution you can trust.

---

## Source Links

- Tinker sampling API docs: https://tinker-docs.thinkingmachines.ai/api-reference/samplingclient
- Tinker types docs: https://tinker-docs.thinkingmachines.ai/api-reference/types
- Async docs: https://tinker-docs.thinkingmachines.ai/async
- Under-the-hood docs: https://tinker-docs.thinkingmachines.ai/under-the-hood
- RL hyperparams docs: https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams
- OpenAI-compatible docs: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai
- Feedback issue #55: https://github.com/thinking-machines-lab/tinker-feedback/issues/55
- Feedback issue #40: https://github.com/thinking-machines-lab/tinker-feedback/issues/40
- Feedback issue #79: https://github.com/thinking-machines-lab/tinker-feedback/issues/79
- Cookbook issue #234: https://github.com/thinking-machines-lab/tinker-cookbook/issues/234
- Cookbook issue #372: https://github.com/thinking-machines-lab/tinker-cookbook/issues/372
