# Debate Recipe: Onboarding Guide

Two LLM debaters argue over GPQA science questions; a frozen LLM judge picks the winner; RL trains the debater to win. The core research question: does optimizing for judge persuasion incentivize truth-telling or judge exploitation?

For architecture details and the full API, see `README.md`. For agent workflow norms, see `AGENTS.md`.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- `TINKER_API_KEY` environment variable
- Network access to HuggingFace (tests load GPQA via `datasets` library)
- WandB (optional): `uv run wandb login`, or pass `wandb_project=""` to disable

## First 30 Minutes

Follow this gate flow top to bottom. Each step must pass before moving on.

```
Preflight: python>=3.12, uv, TINKER_API_KEY
    |
    v
uv sync ────────────────────> fail? fix env/deps first
    |
    v
pytest debate/ ─────────────> fail? core logic broken, stop
    |
    v
dump_io (no API) ───────────> fail? prompt wiring broken
    |
    v
smoke_mcq (live) ───────────> fail? API/network issue
    |
    v
train batch_size=2 group_size=2  (bounded, 2-3 steps)
```

### Step-by-step commands

```bash
# 1. Install dependencies
uv sync

# 2. Offline tests (no API key needed)
uv run pytest tinker_cookbook/recipes/multiplayer_rl/debate/ -x -q

# 3. Dump I/O — verify prompt wiring without API calls
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io \
  --prompts scientific_mcq --protocol sequential
# Check: /tmp/tinker-examples/dump_io.json exists with sensible prompts

# 4. Smoke test — cheapest live API call
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_mcq
# Check: HTML trace at /tmp/tinker-examples/smoke_mcq.html

# 5. Training run — bounded, minimal cost
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  batch_size=2 \
  group_size=2 \
  eval_every=999 \
  save_every=999 \
  wandb_project=""
# Check: stdout shows progress/batch: 0, progress/batch: 1, loss values
# IMPORTANT: eval_every=999 avoids triggering a full GPQA inspect eval each step
```

## Cost & Duration

| Command | Workload | Wall time | Cost |
|---------|----------|-----------|------|
| `pytest debate/` | 380 offline tests | ~25s | $0 |
| `dump_io` | 0 API calls, prints assembled prompts | ~1s | $0 |
| `smoke_mcq` | 2 GPQA problems, group=2, SEQ 2-round, Qwen3-4B | ~3.5min | $0.014 |
| `smoke_judge_exploit` | 1 problem, group=1, SEQ 2-round, judge_exploit, Qwen3-4B | ~3.5min | $0.005 |
| `train` (first batch) | batch=2, group=2, SEQ 2-round, GPQA extended, Qwen3-4B | >10min | ~$0.05+ |

> **NOTE**: Training cold start is slow (~10min for first batch with batch_size=2, group_size=2). This is Tinker API initialization latency, not a bug. Subsequent batches are faster.

## Command to Artifact Map

| Command | Output artifact |
|---------|-----------------|
| `pytest` | pass/fail in terminal |
| `dump_io` | `/tmp/tinker-examples/dump_io.json` |
| `smoke_mcq` | `/tmp/tinker-examples/smoke_mcq.html` |
| `smoke_judge_exploit` | `/tmp/tinker-examples/smoke_judge_exploit.html` |
| `train` | `log_path/` directory (metrics, checkpoints, episodes) |

## Turn Timeline per Protocol

`num_rounds` controls total rounds. `num_rounds=2` means 1 propose round + 1 critique round (round 0 = PROPOSE, rounds 1..N-1 = CRITIQUE).

```
SEQUENTIAL (num_rounds=2):
  slot 0    slot 1      slot 2    slot 3
  A:propose B:propose | A:critique B:critique |
  ────────────────────┼───────────────────────┼──> judge
       round 0        |       round 1         |

SIMULTANEOUS (num_rounds=2):
  slot 0              slot 1
  A+B:propose       | A+B:critique          |
  ──────────────────┼───────────────────────┼──> judge
       round 0      |       round 1         |

HYBRID (num_rounds=2):
  slot 0              slot 1    slot 2
  A+B:propose       | A:critique B:critique |
  ──────────────────┼───────────────────────┼──> judge
       round 0      |       round 1         |

Key: | = round boundary, A+B = simultaneous (buffered commit)
```

## Prompt Resolution Chain

```
prompts_ref="scientific_mcq"
    │
    ▼
resolve_prompts(ref)           prompts/__init__.py:432
    │  LRU-cached YAML load + Jinja2 compile
    ▼
DebatePrompts.render_system(role, phase, ctx)
    │  lookup: phase key, then "default" key, else KeyError
    ▼
build_generation_messages(state, role, trigger)     core/visibility.py:177
    │  system + question + visible transcript + phase instruction + prefill
    ▼
get_field_instructions(role, trigger)               scoring/parsing.py
    │  "You MUST include: <answer>...</answer>"
    ▼
model output → extract_fields(text, specs)          scoring/parsing.py
    │  regex <(\w+)>(.*?)</\1>, coerce types
    ▼
Utterance.fields → trajectory queries → metrics
```

## Prompt Configs

| Config | Used by | Field extraction | Notes |
|--------|---------|------------------|-------|
| `default` | -- | None | Minimal, no structured output |
| `scientific_mcq` | `smoke_mcq` | `<answer>` + `<reasoning>` per debater | GPQA-style MCQ |
| `judge_exploit` | `train.py` default | `<reason>` + `<decision>` on judge, think-tag aware | Research config |

> **IMPORTANT**: `train.py` defaults to `prompts_ref="judge_exploit"`, not `default` or `scientific_mcq`. This is the research config with persuasion emphasis and private `<think>` reasoning. See `scripts/train.py:204`.

## smoke_judge_exploit: Model Access Warning

`smoke_judge_exploit.py` defaults to `openai/gpt-oss-120b` (debater) + `openai/gpt-oss-20b` (judge). Unless you have gpt-oss access, override with accessible models:

```bash
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_judge_exploit \
  --debater-model Qwen/Qwen3-4B-Instruct-2507 \
  --judge-model Qwen/Qwen3-4B-Instruct-2507
```

This script uses `argparse` (not `chz`), so it takes `--flag value` syntax.

## Adding a New Protocol (Checklist)

1. **`types.py`** -- Add variant to `ProtocolKind` enum.

2. **`core/schedule.py`** -- Add `build_schedule()` branch returning your `TurnSlot` sequence. Each slot defines: `actors`, `phase`, `boundary_after`, `visibility_policy`. The `else: raise ValueError` on line 104 catches missing protocols.

3. **Frozen-opponent constraint** (`env.py:356-365`): validates `all_schedule_roles ⊆ {DEBATER_A, DEBATER_B}`. Judge turns are callbacks, not schedule slots. Your protocol must not add judge turns to the schedule.

4. **`prompts/*.yaml`** -- Add phase entries for your protocol's phases. `render_system` raises `KeyError` if neither the phase key nor a `default` key exists. If your protocol uses phases that existing YAML configs don't cover, you'll get a clear error. Verify prompt wiring with:
   ```bash
   uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io \
     --prompts your_template --protocol your_protocol
   ```

5. **Terminology**: `num_rounds=2` means 1 propose round + 1 critique round (round 0 = PROPOSE, rounds 1..N-1 = CRITIQUE). Not 2 full exchanges.

6. **Tests**: Add test_reducer (full playthrough), test_runtime (if simultaneous barrier logic involved), test_integration (mock completers).

7. **Verify**: `uv run pytest tinker_cookbook/recipes/multiplayer_rl/debate/ -x -q`

## CLI Syntax

Training and dataset scripts use `chz.entrypoint`, which requires `key=value` syntax:

```bash
# chz entrypoint (train.py, dataset builders)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  model_name=Qwen/Qwen3-8B \
  batch_size=8 \
  group_size=4 \
  num_rounds=2 \
  prompts_ref=scientific_mcq \
  protocol_kind=SEQUENTIAL
```

All CLI args correspond to fields on `CLIConfig` in `scripts/train.py:188`.

Standalone scripts (`smoke_judge_exploit`, `dump_io`) use `argparse` with `--flag value` syntax:

```bash
# argparse (smoke_judge_exploit, dump_io)
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io \
  --prompts judge_exploit --protocol sequential
```

## Data Flow (Summary)

For the full 11-step walkthrough from GPQA question to trained gradient, see `README.md`. In brief:

1. Load GPQA from HuggingFace, shuffle MCQ options
2. Build `DebateProblem` tuples (task_prompt, answer_a, answer_b, target_label)
3. `DebateDataset` wraps problems, `get_batch()` yields `DebateGroupBuilder` instances
4. `make_envs()` creates `group_size` independent runtimes per problem
5. `build_schedule()` generates `TurnSlot` sequence from protocol + num_rounds
6. Tinker RL loop drives `initial_observation()` / `step(action)`, opponent runs frozen via `MessageCompleter`
7. Field extraction on each `submit()` parses XML tags from model output
8. Judge verdict after schedule exhausts, parsed into `DebateOutcome`
9. Rewards via `outcome_reward_fn` (+1 winner, -1 loser for zero-sum)
10. Metrics remapped from seat-based (debater_a/b) to identity-based (trained/opponent)
11. Advantages computed, `forward_backward_async` with `importance_sampling` loss

## Key Extension Points

### Add a prompt template

1. Create `prompts/my_template.yaml` following the V2 schema (see `prompts/judge_exploit.yaml` as reference).
2. Required sections: `version: 2`, `system` (with `default` key per role), `question` (debater_a + debater_b required).
3. Optional: `user` (phase instructions), `think` (reasoning mode), `prefill`, `fields` (structured extraction).
4. Use it: `prompts_ref=my_template` on CLI.
5. `resolve_prompts()` (`prompts/__init__.py:432`) handles loading, validation, and compilation.

### Add a reward function

Implement `OutcomeRewardFn` (signature: `(DebateOutcome) -> Mapping[Role, float]`):

```python
from tinker_cookbook.recipes.multiplayer_rl.debate.types import DebateOutcome, Role

def accuracy_weighted_reward(outcome: DebateOutcome) -> dict[Role, float]:
    """Winner gets +2 if correct, +0.5 if wrong. Loser gets -1."""
    if outcome.winner is None:
        return {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
    loser = Role.DEBATER_B if outcome.winner == Role.DEBATER_A else Role.DEBATER_A
    return {outcome.winner: 1.0, loser: -1.0}
```

Pass to `DebateDataset(outcome_reward_fn=accuracy_weighted_reward, ...)`.

For step-level rewards, implement `StepRewardFn` (signature: `(before: DebateState, after: DebateState, role: Role, utterance: Utterance | None) -> float`). See `plugins.py` for protocol definitions.
