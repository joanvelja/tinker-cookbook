# Debate Environment

Algorithm-agnostic debate environment for Tinker RL. Two debaters defend candidate answers; a judge evaluates. Works with any training algorithm (GRPO, ExIt, DPO, MCTS, actor-critic).

## Quickstart

```bash
# Setup
export $(cat .env | xargs)   # or set TINKER_API_KEY directly
uv sync

# Smoke test (~$0.01, writes HTML trace)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_mcq

# Train (GRPO on GPQA diamond, frozen opponent + LLM judge)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train

# Train with custom config (chz entrypoint)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  --model_name Qwen/Qwen3-8B \
  --batch_size 8 \
  --group_size 4 \
  --learning_rate 3e-5 \
  --num_rounds 2 \
  --prompts_ref scientific_mcq

# Run tests (offline, no API needed)
uv run pytest tinker_cookbook/recipes/multiplayer_rl/debate/ -x -q
```

The smoke test writes an HTML trace to `/tmp/tinker-examples/smoke_mcq.html` — open it to inspect the full debate (system prompts, per-turn I/O, judge verdict, metrics).

## Architecture

Three layers, strict separation of concerns:

```
Reducer  (Layer 1)  Pure functions on frozen DebateState. No I/O, no async. Forkable.
Runtime  (Layer 2)  Async coordination (Condition, barriers, turn tickets). Renderer-unaware.
Env      (Layer 3)  Thin Tinker adapter (~30 lines). Owns renderer. Converts tokens<->text.
```

**File map:**
```
debate/
  types.py        Frozen types: Role, Phase, TurnSlot, Utterance, DebateState, ...
  plugins.py      StepRewardFn, JudgeCallback, OutcomeRewardFn protocols
  env.py          DebateEnv, DebateGroupBuilder, DebateDataset

  core/
    reducer.py    Pure state transitions (apply_action, commit_slot_actions, ...)
    runtime.py    DebateRuntime: async shell over reducer
    schedule.py   build_schedule() -> tuple[TurnSlot, ...]
    visibility.py Visibility policy registry + get_visible_messages()

  scoring/
    fields.py     FieldSpec, ScoringMode, classifiers, normalizers
    parsing.py    XML field extraction (extract_fields, generate_format_instructions)
    mcq.py        MCQ answer normalization (normalize_mcq, strip_think)
    trajectory.py Transcript queries (final_answer, answers_by_round, ...)
    metrics.py    16 metric factories (accuracy, judge_quality, truth_win, ...)
    judge.py      LLMJudgeCallback with schema-driven verdict parsing

  prompts/
    __init__.py          DebatePrompts, resolve_prompts(), YAML loading
    default.yaml         Minimal debate prompts
    scientific_mcq.yaml  GPQA-style MCQ with field extraction
    galaxy_brain.yaml    Open-ended debate

  scripts/
    train.py       Training loop (GRPO, frozen opponent, LLM judge)
    smoke_mcq.py   Smoke test: 2 GPQA problems, HTML trace
    smoke_galaxy.py  Smoke test: open-ended debate
    dump_io.py     Debug: print assembled prompts for each role/phase
    trace_fmt.py   HTML trace renderer

  tests/            314 offline tests
```

## Protocols

Three debate protocols, selected via `ProtocolKind`:

### Sequential
Debaters alternate: A speaks, then B, forming a round. Both see all prior utterances.

```
round 0:  A proposes  ->  B proposes  [boundary]
round 1:  A critiques ->  B critiques [boundary]
```

### Simultaneous
Both debaters act in parallel per round. During composition, each sees only completed rounds (not the opponent's current-round text).

```
round 0:  A,B propose simultaneously  [boundary]
round 1:  A,B critique simultaneously [boundary]
```

### Hybrid
Round 0 is simultaneous (blind proposals), subsequent rounds are sequential (informed critiques).

```
round 0:  A,B propose simultaneously  [boundary]
round 1:  A critiques -> B critiques  [boundary]
```

With `include_judge_turns=True`, `JUDGE_QUERY` and `JUDGE_VERDICT` slots are appended after each round boundary.

## Usage

### Training config

`train.py` uses `chz` for config. Key knobs in `CLIConfig`:

| Param | Default | What it does |
|---|---|---|
| `model_name` | `Qwen/Qwen3-4B-Instruct-2507` | Trained debater model |
| `opponent_model` | same | Frozen opponent model |
| `judge_model` | same | LLM judge model |
| `batch_size` | 4 | Problems per training batch |
| `group_size` | 4 | Rollouts per problem (for advantage estimation) |
| `num_rounds` | 2 | Debate rounds (propose + N-1 critique rounds) |
| `learning_rate` | 3e-5 | LoRA learning rate |
| `prompts_ref` | `default` | Prompt config (`default`, `scientific_mcq`, `galaxy_brain`) |
| `protocol_kind` | `SEQUENTIAL` | `SEQUENTIAL`, `SIMULTANEOUS`, or `HYBRID` |
| `randomize_position` | `True` | Trained model plays both debater_a and debater_b |

### Programmatic usage

```python
from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateDataset

dataset = DebateDataset(
    problems=[("Is P=NP?", "Yes", "No", "No")],  # (prompt, ans_a, ans_b, target)
    batch_size=1,
    renderer=renderer,
    protocol_kind=ProtocolKind.SEQUENTIAL,
    num_rounds=2,
    judge_callback=judge_callback,
    outcome_reward_fn=zero_sum_outcome_reward,
    opponent_completer=opponent_completer,
    group_size=4,
    prompts_ref="scientific_mcq",
    metrics=mcq_debate_metrics(),
)
```

Each `get_batch()` returns `DebateGroupBuilder` instances that produce paired `DebateEnv` objects sharing a runtime.

### Custom rewards with plugins

```python
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome, DebateState, Role, Utterance,
)

# Step reward: penalize verbose responses
def brevity_reward(before: DebateState, after: DebateState,
                   role: Role, utterance: Utterance | None) -> float:
    if utterance is None:
        return 0.0
    return -0.001 * utterance.token_count

# Outcome reward: winner gets +1, loser gets -1
def zero_sum(outcome: DebateOutcome) -> dict[Role, float]:
    if outcome.winner is None:
        return {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
    loser = Role.DEBATER_B if outcome.winner == Role.DEBATER_A else Role.DEBATER_A
    return {outcome.winner: 1.0, loser: -1.0}

from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateGroupBuilder

group = DebateGroupBuilder(
    task_prompt="Is P=NP?",
    answer_a="Yes",
    answer_b="No",
    renderer=renderer,
    protocol_kind="sequential",
    num_rounds=2,
    step_reward_fn=brevity_reward,
    outcome_reward_fn=zero_sum,
)
```

### Judge callback

```python
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome, JudgeDecision, JudgeRequest, Role,
)

class LLMJudge:
    async def on_boundary(self, request: JudgeRequest) -> JudgeDecision | None:
        # Score after each round (optional)
        return JudgeDecision(
            round_index=request.state.rounds_completed - 1,
            verdict="A is more convincing",
            score_delta_by_role={Role.DEBATER_A: 0.5, Role.DEBATER_B: -0.5},
        )

    async def on_final(self, request: JudgeRequest) -> DebateOutcome:
        # Required: produce final outcome
        return DebateOutcome(
            winner=Role.DEBATER_A,
            scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: 0.0},
            verdict_text="A provided stronger arguments.",
        )
```

### Forking / branching (MCTS, tree search)

Capture a snapshot mid-debate, then branch into independent continuations:

```python
from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateBranchGroupBuilder

# Capture state at any point
snapshot = runtime.snapshot(
    renderer_name="qwen3",
    protocol_kind="sequential",
    protocol_kwargs={},
)

# Create N independent branches from the same state
branches = [
    DebateBranchGroupBuilder(
        snapshot=snapshot,
        renderer=renderer,
    )
    for _ in range(n_branches)
]
```

Each branch gets an independent `DebateRuntime` with a forked copy of the state. State is frozen/immutable, so forking is free.

## Plugin API

| Protocol | Signature | When called |
|---|---|---|
| `StepRewardFn` | `(before, after, role, utterance) -> float` | After each action commit |
| `JudgeCallback.on_boundary` | `(JudgeRequest) -> JudgeDecision \| None` | After each round boundary |
| `JudgeCallback.on_final` | `(JudgeRequest) -> DebateOutcome` | When schedule is exhausted |
| `OutcomeRewardFn` | `(DebateOutcome) -> Mapping[Role, float]` | In `compute_group_rewards()` |

## Design notes

Text is canonical in reducer/runtime layers; token conversion happens only in the env layer. All state is frozen (`DebateState` is immutable; transitions produce new instances), which makes forking trivial.

Visibility is policy-based: each `TurnSlot` has a `visibility_policy` that keys into a registry of filter functions (`all_prior` or `completed_rounds_only`). When `open_reasoning=False`, `<thinking>` tags are stripped from opponent messages.

In simultaneous slots, the first arriver buffers and waits on an `asyncio.Condition`; the last arriver commits and notifies all waiters.

## Scoring pipeline

YAML field definitions → `FieldSpec` with scoring + normalizer → XML extraction from model output → structured fields on `Utterance` → transcript queries → metrics + rewards.

```
Model output: "<answer>C</answer><reasoning>because...</reasoning>"
    ↓ extract_fields(text, field_specs)         [parsing.py]
    ↓ stored on Utterance.fields                [reducer.py, runtime.py]
    ↓ queried via final_answer(), answers_by_round()  [trajectory.py]
    ↓ metric computation                        [metrics.py]
accuracy(DEBATER_A)(state) → MetricResult(1.0)
```

Field specs are defined in YAML prompt configs (e.g. `scientific_mcq.yaml`):
```yaml
fields:
  debater_a:
    propose:
      answer: {type: str, scoring: {mode: enum, values: [A, B, C, D]}}
      reasoning: {type: str, description: "your scientific reasoning"}
```

### Key metrics

| Metric | What it measures |
|---|---|
| `truth_win_if_disagreement` | Does debate help judges pick truth under adversarial pressure? Primary alignment signal. |
| `judge_quality` | Did the debate produce the right verdict? |
| `truth_surfaced` | Was the correct answer argued by any debater? |
| `concession_correctness` | +1 revised from wrong, -1 capitulated from correct. Detects sycophancy. |
| `accuracy(role)` | Per-role final answer correctness. |
| `disagreement` | Did adversarial pressure manifest? |
| `draw_rate` | Is the judge hedging? |

See `metrics.py` for all 16 metrics and `mcq_debate_metrics()` for the default set.

## Testing

All 314 tests run offline.

```bash
uv run pytest tinker_cookbook/recipes/multiplayer_rl/debate/ -x -q
uv run pytest tinker_cookbook/recipes/multiplayer_rl/debate/test_reducer.py -x -q

# Debug prompt assembly (prints what each role sees at each phase)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io --prompts scientific_mcq
```

## References

- Irving, G., Christiano, P., & Amodei, D. (2018). AI safety via debate. arXiv:1805.00899.
- Khan, A., et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. ICML 2024.
