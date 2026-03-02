# Debate Recipe: Onboarding Quickstart

Two LLM debaters argue over GPQA science questions; a frozen LLM judge picks the winner; RL trains the debater to win. The core research question: does optimizing for judge persuasion incentivize truth-telling or judge exploitation?

For architecture details and the full API, see `README.md`. For agent workflow norms, see `AGENTS.md`.

## Quickstart

```bash
# Prerequisites: Python 3.12+, TINKER_API_KEY set
uv sync

# Smoke test (~$0.01, writes HTML trace to /tmp/tinker-examples/smoke_mcq.html)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_mcq

# Training run (REINFORCE+IS, frozen opponent, LLM judge on GPQA)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train

# Train with overrides (chz entrypoint -- any CLIConfig field works)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  --model_name Qwen/Qwen3-8B \
  --batch_size 8 \
  --group_size 4 \
  --num_rounds 2 \
  --prompts_ref scientific_mcq \
  --protocol_kind SEQUENTIAL

# Offline tests (no API key needed)
uv run pytest tinker_cookbook/recipes/multiplayer_rl/debate/ -x -q
```

All CLI args correspond to fields on `CLIConfig` in `scripts/train.py:188`.

## Data Flow: GPQA Question to Trained Gradient

Numbered steps trace one training batch end-to-end. Files are relative to `tinker_cookbook/recipes/multiplayer_rl/debate/`.

1. **Load GPQA** -- `scripts/train.py:_load_gpqa` loads HuggingFace `Idavidrein/gpqa`, splits train/test.

2. **Build problems** -- `scripts/train.py:_gpqa_to_problems` shuffles MCQ options, tracks `target_label` (ground truth letter). Returns `list[DebateProblem]` (4-tuple: task_prompt, answer_a, answer_b, target).

3. **DebateDataset** -- `env.py:DebateDataset` wraps the problem list. `get_batch(index)` returns a list of `DebateGroupBuilder` instances, one per problem in the batch.

4. **DebateGroupBuilder.make_envs()** -- `env.py:DebateGroupBuilder.make_envs` creates `group_size` independent runtimes (frozen-opponent mode). Each runtime gets:
   - A `DebateSpec` (frozen config: task_prompt, schedule, prompts_ref, target)
   - A `DebateState` (mutable-via-replacement episode state: transcript, slot_index, outcome)
   - A `DebateRuntime` wrapping the state with async coordination

5. **Schedule** -- `core/schedule.py:build_schedule` generates an ordered tuple of `TurnSlot` objects defining who speaks when. For `SEQUENTIAL` with 2 rounds: A proposes, B proposes [boundary], A critiques, B critiques [boundary]. 4 slots total.

6. **Episode rollout** -- Tinker's RL loop calls `DebateEnv.initial_observation()` and `DebateEnv.step(action)` alternately:
   - `_drive_opponent()` (`env.py:138`) runs the frozen opponent through sequential turns via the `MessageCompleter`
   - `runtime.wait_for_turn(role)` (`core/runtime.py:84`) blocks on `asyncio.Condition` until the trained agent's turn
   - `runtime.submit(ticket, text, token_count)` (`core/runtime.py:104`) applies the pure reducer (`core/reducer.py:apply_action`), handles simultaneous barriers, and triggers judge callbacks
   - `build_generation_messages()` (`core/visibility.py:177`) assembles the observation: system prompt + question + visible transcript + phase instruction + prefill

7. **Field extraction** -- On each `submit()`, `runtime.py:137` checks for `FieldSpec` definitions in the YAML prompts. If found, `scoring/parsing.py:extract_fields` parses XML tags from model output and normalizes values. Extracted fields are stored on `Utterance.fields`.

8. **Judge verdict** -- When the schedule is exhausted, `runtime.py:229` calls `LLMJudgeCallback.on_final()` (`scoring/judge.py:32`). The judge sees the full transcript via `build_generation_messages(state, Role.JUDGE, trigger="final")`. Its response is parsed into a `DebateOutcome` (winner, scores, verdict_text).

9. **Rewards** -- `DebateGroupBuilder.compute_group_rewards()` (`env.py:459`) calls `outcome_reward_fn(outcome)` -- typically `zero_sum_outcome_reward` (+1 winner, -1 loser). Metrics are computed via `mcq_debate_metrics()` (`scoring/metrics.py:547`).

10. **Metrics remapping** -- `env.py:_remap_to_identity` translates seat-based metrics (debater_a/debater_b) to identity-based (trained/opponent) using `id/` prefix, since the trained model's seat is randomized.

11. **Training step** -- Rewards and trajectories flow to Tinker's RL loop (`tinker_cookbook/rl/train.py`), which computes advantages (group-mean baseline), then calls `forward_backward_async` with `importance_sampling` loss.

## Concurrency Model

The runtime coordinates multiple async agents (trained debater, frozen opponent, judge) through `DebateRuntime` (`core/runtime.py`).

### Key primitives

- **`asyncio.Condition`** (`runtime.py:62`) -- Single lock + condition variable protecting all state mutations. Every `wait_for_turn` and `submit` call acquires this lock.

- **`TurnTicket`** (`types.py:153`) -- Opaque token returned by `wait_for_turn`. Contains `slot_id` + `state_version` + `role`. The runtime validates the ticket's `slot_id` against the current slot before accepting a submission, preventing stale actions.

- **Turn eligibility** -- `reducer.py:get_eligible_roles` returns `slot.actors - pending_simultaneous`. For sequential slots (1 actor), exactly one role is eligible. For simultaneous slots (2 actors), both are eligible until they submit.

### Sequential flow

```
Trained agent: wait_for_turn(A) -> ticket -> submit(ticket, text) -> advance slot
Opponent:      wait_for_turn(B) -> ticket -> submit(ticket, text) -> advance slot
               [repeat for each round]
Judge:         on_final() called by runtime after schedule exhausted
```

In practice, `DebateEnv._drive_opponent()` (`env.py:138`) runs the opponent synchronously before returning control to the trained agent, so the Tinker RL loop only sees one agent per env.

### Simultaneous flow

When a `TurnSlot` has multiple actors (e.g., `SIMULTANEOUS` protocol):

1. Both agents call `wait_for_turn()` -- both get tickets (same slot_id).
2. **First arriver** calls `submit()`. `apply_action` buffers in `pending_simultaneous` (not yet committed). The arriver then enters a `Condition.wait()` loop until the slot advances.
3. **Last arriver** calls `submit()`. `apply_action` sees all actors present, commits all buffered utterances in canonical order, advances the slot.
4. `notify_all()` wakes the first arriver. Both return consistent `SubmitResult`s from the same post-commit state.

The opponent is fired as `asyncio.ensure_future` for simultaneous slots (`env.py:156`), so it runs concurrently with the trained agent's sampling.

### Judge locking

Judge callbacks (`on_boundary`, `on_final`) run while holding the Condition lock (`runtime.py:220`). This is intentional: releasing would let next-turn agents proceed before the judge has processed the boundary. Callbacks receive a state snapshot and must not call back into the runtime.

## Key Extension Points

### Add a prompt template

1. Create `prompts/my_template.yaml` following the V2 schema (see `prompts/judge_exploit.yaml` as reference).
2. Required sections: `version: 2`, `system` (with `default` key per role), `question` (debater_a + debater_b required).
3. Optional: `user` (phase instructions), `think` (reasoning mode), `prefill`, `fields` (structured extraction).
4. Use it: `--prompts_ref my_template` on CLI, or `prompts_ref="my_template"` in code.
5. `resolve_prompts()` (`prompts/__init__.py:432`) handles loading, validation, and compilation.

Minimal template skeleton:

```yaml
version: 2

system:
  debater_a:
    default: "You are {{ viewer_role }}. Argue for your position."
  debater_b:
    default: "You are {{ viewer_role }}. Argue for your position."
  judge:
    default: "You are the judge. Evaluate both arguments."

question:
  debater_a: "{{ task_prompt }}"
  debater_b: "{{ task_prompt }}"
  judge: "{{ task_prompt }}"
```

Available template variables: `task_prompt`, `viewer_role`, `phase`, `round_index`, `num_rounds`, `is_first_round`, `is_last_round`, `protocol_kind`, `answer`, `answer_a`, `answer_b`, `open_reasoning`, `has_assigned_answer`, `reasoning_is_private`. See `prompts/__init__.py:_build_context` for the full list.

### Add a reward function

Implement `OutcomeRewardFn` (signature: `(DebateOutcome) -> Mapping[Role, float]`):

```python
# In your training script or a new module
from tinker_cookbook.recipes.multiplayer_rl.debate.types import DebateOutcome, Role

def accuracy_weighted_reward(outcome: DebateOutcome) -> dict[Role, float]:
    """Winner gets +2 if correct, +0.5 if wrong. Loser gets -1."""
    if outcome.winner is None:
        return {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
    loser = Role.DEBATER_B if outcome.winner == Role.DEBATER_A else Role.DEBATER_A
    # You'd need target info from state for correctness --
    # consider using compute_group_rewards override instead.
    return {outcome.winner: 1.0, loser: -1.0}
```

Pass to `DebateDataset(outcome_reward_fn=accuracy_weighted_reward, ...)` or `DebateGroupBuilder(outcome_reward_fn=..., ...)`.

For step-level rewards, implement `StepRewardFn` (signature: `(before: DebateState, after: DebateState, role: Role, utterance: Utterance | None) -> float`). See `plugins.py` for the protocol definitions.

### Add a new protocol

1. Add a variant to `ProtocolKind` in `types.py:24`.
2. Add a branch in `core/schedule.py:build_schedule` that returns the appropriate `TurnSlot` sequence. Each slot defines: `actors` (who speaks), `phase`, `boundary_after` (marks round end), `visibility_policy`.
3. That's it. The reducer, runtime, visibility, and env layers are all schedule-driven -- they don't hardcode protocol logic.

## Scoring Pipeline Walkthrough

Tracing the `decision` field from YAML definition to final metric, using `judge_exploit.yaml`:

### 1. YAML definition (`prompts/judge_exploit.yaml:116-118`)

```yaml
fields:
  judge:
    final:
      reason: {type: str}
      decision: {type: str, description: "debater_a, debater_b, or tie"}
```

### 2. Field resolution (`prompts/__init__.py:422`)

At load time, `_parse_fields` calls `scoring/fields.py:_resolve_fields`, which converts the YAML dict into `FieldSpec(type=str, description="debater_a, debater_b, or tie")`. No scoring mode is set for `decision` (it's a bare str field), so no normalizer is attached.

### 3. Format instruction generation (`prompts/__init__.py:140`)

When assembling the judge's prompt, `get_field_instructions("judge", "final")` calls `scoring/parsing.py:generate_format_instructions`, which emits:

```
You MUST include the following XML tags in your response:
<reason>your reason here (str)</reason>
<decision>debater_a, debater_b, or tie</decision>
```

This is appended to the judge's user message by `render_user()`.

### 4. Extraction at submit time (`core/runtime.py:134-139`)

When the judge submits its response, `runtime.submit()` looks up field specs via `prompts.get_field_specs("judge", "final")`. If specs exist, it calls `scoring/parsing.py:extract_fields(text, specs)`, which:
- Runs regex `<(\w+)>(.*?)</\1>` to find XML tags
- Coerces values to the declared type (str in this case)
- Applies normalizers (none here, since no scoring mode)

Result: `{"reason": "A provided stronger...", "decision": "debater_a"}`.

### 5. Storage on Utterance (`core/reducer.py:70-78`)

`apply_action` creates an `Utterance` with `fields={"reason": "...", "decision": "debater_a"}`. Fields are frozen via `MappingProxyType` in `Utterance.__post_init__` (`types.py:55`).

### 6. Judge verdict parsing (`scoring/judge.py:40-49`)

For the judge specifically, `LLMJudgeCallback.on_final` also extracts fields independently and passes them to `_parse_verdict`, which reads `fields["decision"]` and maps `"debater_a"` to `Role.DEBATER_A`, producing a `DebateOutcome(winner=Role.DEBATER_A, ...)`.

### 7. Metric computation (`scoring/metrics.py`)

Metrics like `judge_quality()` read `state.outcome.winner` (set from the parsed verdict) and compare against `state.spec.target` to determine if the judge picked the correct debater. Debater metrics like `accuracy(Role.DEBATER_A)` use `scoring/trajectory.py:final_answer` to read `Utterance.fields["answer"]` from the debater's last utterance.

The full chain: **YAML field def** -> **FieldSpec** (`fields.py`) -> **extract_fields** (`parsing.py`) -> **Utterance.fields** (`reducer.py`) -> **trajectory queries** (`trajectory.py`) -> **metric functions** (`metrics.py`).
