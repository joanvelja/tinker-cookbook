# Debate Environment

Algorithm-agnostic debate environment for Tinker RL. Two debaters defend candidate answers; a judge evaluates. Produces structured trajectories consumable by any training algorithm (GRPO, ExIt, DPO, MCTS, actor-critic).

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
  types.py       Frozen types: Role, Phase, TurnSlot, Utterance, DebateState, ...
  plugins.py     StepRewardFn, JudgeCallback, OutcomeRewardFn protocols
  schedule.py    build_schedule() -> tuple[TurnSlot, ...]
  visibility.py  Visibility policy registry + get_visible_messages()
  reducer.py     Pure state transitions (apply_action, commit_slot_actions, ...)
  runtime.py     DebateRuntime: async shell over reducer
  env.py         DebateEnv, DebateGroupBuilder, DebateDataset, DebateDatasetBuilder
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

### Basic GRPO training

```python
from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateDatasetBuilder

builder = DebateDatasetBuilder(
    model_name="Qwen/Qwen3-8B",
    renderer_name="qwen3",
    protocol_kind="sequential",
    num_rounds=2,
    train_problems=[
        ("Is P=NP?", "Yes, P=NP", "No, P!=NP"),
        ("Is the Earth flat?", "Yes", "No"),
    ],
)
train_dataset, test_dataset = await builder()
```

Pass `train_dataset` to any Tinker RL training loop (GRPO, etc.). Each `get_batch()` returns `DebateGroupBuilder` instances that produce paired `DebateEnv` objects sharing a runtime.

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

## Key design decisions

- **Text is canonical** in reducer/runtime layers. Token conversion happens only in the env layer.
- **All state is frozen.** `DebateState` is an immutable dataclass; transitions produce new instances. This makes forking trivial and eliminates mutation bugs.
- **Visibility is policy-based.** The `visibility_policy` field on each `TurnSlot` keys into a registry of filter functions. Two built-in policies: `all_prior` (see everything) and `completed_rounds_only` (hide current-round opponent text).
- **Reasoning stripping.** When `open_reasoning=False`, `<thinking>...</thinking>` tags are stripped from opponent messages in the visibility layer.
- **Simultaneous barrier.** In simultaneous slots, the first arriver buffers and waits on an `asyncio.Condition`; the last arriver commits and notifies all waiters.

## References

- Irving, G., Christiano, P., & Amodei, D. (2018). AI safety via debate. arXiv:1805.00899.
- Khan, A., et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. ICML 2024.
