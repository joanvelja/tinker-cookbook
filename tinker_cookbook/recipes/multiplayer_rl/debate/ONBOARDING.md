# Debate Recipe: Onboarding Guide

>*"Two LLM debaters argue over questions; a frozen LLM judge picks the winner; RL trains the debater to win. RQ: does optimizing for judge persuasion incentivize truth-telling or judge exploitation?"*

Architecture details and the full API live in `README.md`. Agent workflow norms are in `AGENTS.md`.

## Prerequisites

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and a `TINKER_API_KEY` env var. Tests pull GPQA from HuggingFace via the `datasets` library, so you need network access. WandB is optional: `uv run wandb login` to enable, or pass `wandb_project=""` to skip.

## First 30 Minutes

Gate flow. Each step must pass before moving on.

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

# 3. Dump I/O: verify prompt wiring without API calls
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io \
  --prompts scientific_mcq --protocol sequential
# Check: /tmp/tinker-examples/dump_io.json exists with sensible prompts

# 4. Smoke test: cheapest live API call
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_mcq
# Check: HTML trace at /tmp/tinker-examples/smoke_mcq.html

# 5. Training run: bounded, minimal cost
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

Training cold start takes ~10min for the first batch even with batch_size=2. This is Tinker API initialization latency (subsequent batches are faster).

## Command to Artifact Map

| Command | Output artifact |
|---------|-----------------|
| `pytest` | pass/fail in terminal |
| `dump_io` | `/tmp/tinker-examples/dump_io.json` |
| `smoke_mcq` | `/tmp/tinker-examples/smoke_mcq.html` |
| `smoke_judge_exploit` | `/tmp/tinker-examples/smoke_judge_exploit.html` |
| `train` | `log_path/` directory (metrics, checkpoints, episodes) |

## Turn Timeline per Protocol

`num_rounds` controls total rounds. `num_rounds=2` = 1 propose round + 1 critique round (round 0 = PROPOSE, rounds 1..N-1 = CRITIQUE).

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
    │  strict templates[phase] lookup, KeyError if missing
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
Utterance.fields → field queries → metrics
```

## Prompt Configs

| Config | Used by | Field extraction | Notes |
|--------|---------|------------------|-------|
| `default` | -- | None | Minimal, no structured output |
| `scientific_mcq` | `smoke_mcq` | `<answer>` + `<reasoning>` per debater | GPQA-style MCQ |
| `judge_exploit` | `train.py` default | `<reason>` + `<decision>` on judge, think-tag aware | Research config |

> **IMPORTANT**: `train.py` defaults to `prompts_ref="judge_exploit"`. This is the research config with persuasion emphasis and private `<think>` reasoning. See `scripts/train.py:204`.

## smoke_judge_exploit: Model Access Warning

`smoke_judge_exploit.py` defaults to `openai/gpt-oss-120b` (debater) + `openai/gpt-oss-20b` (judge). Unless you have gpt-oss access, override:

```bash
uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_judge_exploit \
  --debater-model Qwen/Qwen3-4B-Instruct-2507 \
  --judge-model Qwen/Qwen3-4B-Instruct-2507
```

This script uses `argparse`, so it takes `--flag value` syntax (unlike `chz` scripts).

## Adding a New Protocol (Checklist)

1. **`types.py`**: Add variant to `ProtocolKind` enum.

2. **`core/schedule.py`**: Add `build_schedule()` branch returning your `TurnSlot` sequence. Each slot defines: `actors`, `phase`, `boundary_after`, `visibility_policy`. The `else: raise ValueError` on line 104 catches missing protocols.

3. **Frozen-opponent constraint** (`env.py:356-365`): validates `all_schedule_roles ⊆ {DEBATER_A, DEBATER_B}`. Judge turns are callbacks (they never appear as schedule slots). Your protocol must not add judge turns to the schedule.

4. **`prompts/*.yaml`**: Add phase entries for your protocol's phases. `'default'` gets copied to every phase at load time, so there's no runtime fallback. Mixing `default` with phase-specific keys raises `ValueError`. Either use `default` alone (same for all phases) or list every phase explicitly. Verify prompt wiring:
   ```bash
   uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io \
     --prompts your_template --protocol your_protocol
   ```

5. **Terminology**: `num_rounds=2` = 1 propose round + 1 critique round (round 0 = PROPOSE, rounds 1..N-1 = CRITIQUE).

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

## Data Flow

Full 11-step walkthrough lives in `README.md`. The short version:

GPQA questions load from HuggingFace with shuffled MCQ options, then get wrapped as `DebateProblem` tuples. `DebateDataset.get_batch()` yields `DebateGroupBuilder` instances, each creating `group_size` independent runtimes via `make_envs()`. The schedule (`build_schedule()`) produces a `TurnSlot` sequence from protocol + num_rounds.

The Tinker RL loop drives `initial_observation()` / `step(action)` while the frozen opponent runs via `MessageCompleter`. On each `submit()`, field extraction parses XML tags from model output. After the schedule exhausts, the judge renders a verdict parsed into `DebateOutcome`.

Rewards come from `outcome_reward_fn` (+1 winner, -1 loser for zero-sum). Metrics get remapped from seat-based (debater_a/b) to identity-based (trained/opponent). Finally, advantages are computed and fed to `forward_backward_async` with `importance_sampling` loss.

## Key Extension Points

### Add a prompt template

Create `prompts/my_template.yaml` following the V2 schema (see `prompts/judge_exploit.yaml` as reference). Required sections: `version: 2`, `system` (with a key per role), `question` (debater_a + debater_b required). Optional sections: `user` (phase instructions), `think` (reasoning mode), `prefill`, `fields` (structured extraction). Pass it as `prompts_ref=my_template` on CLI. `resolve_prompts()` (`prompts/__init__.py:432`) handles loading, validation, and compilation.

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
