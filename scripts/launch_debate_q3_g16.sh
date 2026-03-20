#!/bin/bash
# Debate self-play GPQA — Q3-30B-A3B-Instruct, G=16, simultaneous, 3 seeds
# Same-model judge via OpenRouter/Alibaba (no-think)
#
# Config: mirrors RLVR/consultancy hparams where applicable
#   - self_play=True, protocol=simultaneous, 2 rounds
#   - judge: qwen3-30b-a3b via OpenRouter (reasoning_effort=none)
#   - prompts: open_selfplay_simultaneous
#
# Launch: bash scripts/launch_debate_q3_g16.sh
# Single: bash scripts/launch_debate_q3_g16.sh 42
#
# Monitor: uv run python -m scripts.spot_check logs/debate-q3-g16/* --watch

set -euo pipefail

LOG_DIR="logs/debate-q3-g16"
mkdir -p "$LOG_DIR"

COMMON=(
  model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
  renderer_name=qwen3_instruct
  self_play=True
  protocol_kind=simultaneous
  num_rounds=2
  prompts_ref=open_selfplay_simultaneous
  problem_source=GPQAOpenEndedProblemSource
  problem_source.subset=extended

  # Answer-correctness scorer (gpt-5-mini via OpenAI API, same as RLVR)
  scorer_builder.provider=openai_compatible
  scorer_builder.model=gpt-5-mini
  scorer_builder.api_key_env=OPENAI_API_KEY
  scorer_builder.reasoning_effort=medium

  # Same-model judge via OpenRouter
  judge_model=qwen/qwen3-30b-a3b
  judge_provider=openai_compatible
  judge_base_url=https://openrouter.ai/api/v1
  judge_api_key_env=OPENROUTER_API_KEY
  judge_reasoning_effort=none
  judge_temperature=0.7
  judge_top_p=0.8
  judge_top_k=20
  judge_max_tokens=2048

  # Format/EoS penalty + reward gating (matches RLVR semantics)
  format_penalty=True
  gate_reward_on_format=True

  # Hparams (full parity with RLVR/consultancy)
  batch_size=64
  group_size=16
  max_tokens=8192
  learning_rate=1e-5
  temperature=1.0
  top_p=0.8
  top_k=20
  eval_temperature=0.7
  eval_top_p=0.8
  eval_top_k=20
  loss_fn=ppo
  clip_ratio_upper=0.28
  advantage_scheme=power_mean
  advantage_alpha=0.5
  kl_penalty_coef=0.0
  normalize_advantages_by_length=True
  grad_clip_norm=0.0
  num_substeps=2
  eval_every=4
  save_every=4
  wandb_project=debate-gpqa-main-fig
  behavior_if_log_dir_exists=resume
)

IDX=0
run_one() {
  local SEED=$1
  local NAME="debate-q3-g16-s${SEED}"
  local DELAY=$((IDX * 90))
  echo "=== Launching ${NAME} (delay=${DELAY}s) ==="
  uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train \
    "${COMMON[@]}" \
    problem_source.seed="${SEED}" \
    log_path="${LOG_DIR}/${NAME}" \
    wandb_name="${NAME}" \
    > "${LOG_DIR}/${NAME}.stdout.log" 2>&1 &
  echo "  PID: $!  (log: ${LOG_DIR}/${NAME}.stdout.log)"
  IDX=$((IDX + 1))
}

# Single run mode
if [[ $# -eq 1 ]]; then
  run_one "$1"
  wait
  exit 0
fi

# Full: 3 seeds
PIDS=()
for SEED in 42 137 7; do
  run_one "$SEED"
  PIDS+=($!)
done

echo ""
echo "=== All 3 launched ==="
echo "  PIDs: ${PIDS[*]}"
echo ""
echo "Monitor:"
echo "  uv run python -m scripts.spot_check ${LOG_DIR}/* --watch"
echo ""
echo "Kill all:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Tail a run:"
echo "  tail -f ${LOG_DIR}/debate-q3-g16-s42.stdout.log"
echo ""

FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=$((FAILED + 1))
done
if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/3 runs failed"
  exit 1
fi
echo "All 3 runs completed successfully."
