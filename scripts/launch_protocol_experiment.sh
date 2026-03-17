#!/bin/bash
# Protocol comparison experiment — 4 runs
# Launch from repo root: bash scripts/launch_protocol_experiment.sh
#
# Runs:
#   1. seq-v1:        SEQUENTIAL + old judge prompt (control)
#   2. seq-v2:        SEQUENTIAL + protocol-aware judge
#   3. hybrid:        HYBRID + proposal-primary judge
#   4. simultaneous:  SIMULTANEOUS + two-cases-plus-audits judge
#
# All share: Qwen3.5-27B (non-thinking), self-play, GPQA open-ended, num_minibatches=8
#
# Monitor: uv run python -m scripts.spot_check logs/protocol-experiment/* --watch

set -euo pipefail

COMMON=(
  model_name=Qwen/Qwen3.5-27B
  renderer_name=qwen3_5_disable_thinking
  opponent_model=Qwen/Qwen3.5-27B
  judge_model=Qwen/Qwen3.5-27B
  judge_renderer_name=qwen3_5_disable_thinking
  self_play=True
  problem_source=GPQAOpenEndedProblemSource
  scorer_builder.provider=openai_compatible
  scorer_builder.model=gpt-5-mini
  scorer_builder.reasoning_effort=medium
  scorer_builder.timeout_s=120
  batch_size=32
  group_size=4
  n_epochs=3
  num_rounds=2
  max_tokens=8192
  num_minibatches=8
  format_penalty=True
  max_connections=1024
  eval_every=5
  save_every=5
  wandb_project=debate-protocol-comparison
)

# 1. SEQUENTIAL + old judge (control)
echo "=== Launching seq-v1 (control) ==="
uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  "${COMMON[@]}" \
  protocol_kind=sequential \
  prompts_ref=open_selfplay \
  log_path=logs/protocol-experiment/seq-v1 \
  wandb_name=seq-v1-control \
  2>&1 | tee logs/protocol-experiment/seq-v1.stdout.log &
PID_SEQ_V1=$!
echo "  PID: $PID_SEQ_V1"

# 2. SEQUENTIAL + protocol-aware judge
echo "=== Launching seq-v2 ==="
uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  "${COMMON[@]}" \
  protocol_kind=sequential \
  prompts_ref=open_selfplay_v2 \
  log_path=logs/protocol-experiment/seq-v2 \
  wandb_name=seq-v2-aware \
  2>&1 | tee logs/protocol-experiment/seq-v2.stdout.log &
PID_SEQ_V2=$!
echo "  PID: $PID_SEQ_V2"

# 3. HYBRID + proposal-primary judge
echo "=== Launching hybrid ==="
uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  "${COMMON[@]}" \
  protocol_kind=hybrid \
  prompts_ref=open_selfplay_hybrid \
  log_path=logs/protocol-experiment/hybrid \
  wandb_name=hybrid \
  2>&1 | tee logs/protocol-experiment/hybrid.stdout.log &
PID_HYBRID=$!
echo "  PID: $PID_HYBRID"

# 4. SIMULTANEOUS + two-cases judge
echo "=== Launching simultaneous ==="
uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  "${COMMON[@]}" \
  protocol_kind=simultaneous \
  prompts_ref=open_selfplay_simultaneous \
  log_path=logs/protocol-experiment/simultaneous \
  wandb_name=simultaneous \
  2>&1 | tee logs/protocol-experiment/simultaneous.stdout.log &
PID_SIMUL=$!
echo "  PID: $PID_SIMUL"

echo ""
echo "=== All 4 launched ==="
echo "  seq-v1 (control): PID $PID_SEQ_V1"
echo "  seq-v2 (aware):   PID $PID_SEQ_V2"
echo "  hybrid:           PID $PID_HYBRID"
echo "  simultaneous:     PID $PID_SIMUL"
echo ""
echo "Monitor:"
echo "  uv run python -m scripts.spot_check logs/protocol-experiment/* --watch"
echo ""
echo "Kill all:"
echo "  kill $PID_SEQ_V1 $PID_SEQ_V2 $PID_HYBRID $PID_SIMUL"
