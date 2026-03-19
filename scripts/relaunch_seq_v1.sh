#!/bin/bash
set -euo pipefail
rm -rf logs/protocol-experiment/seq-v1
exec uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
  model_name=Qwen/Qwen3.5-27B \
  renderer_name=qwen3_5_disable_thinking \
  opponent_model=Qwen/Qwen3.5-27B \
  judge_model=Qwen/Qwen3.5-27B \
  judge_renderer_name=qwen3_5_disable_thinking \
  self_play=True \
  problem_source=GPQAOpenEndedProblemSource \
  scorer_builder.provider=openai_compatible \
  scorer_builder.model=gpt-5-mini \
  scorer_builder.reasoning_effort=medium \
  scorer_builder.timeout_s=120 \
  batch_size=32 \
  group_size=4 \
  n_epochs=3 \
  num_rounds=2 \
  max_tokens=8192 \
  num_minibatches=8 \
  format_penalty=True \
  max_connections=1024 \
  eval_every=5 \
  save_every=5 \
  wandb_project=debate-protocol-comparison \
  protocol_kind=sequential \
  prompts_ref=open_selfplay \
  log_path=logs/protocol-experiment/seq-v1 \
  wandb_name=seq-v1-control
