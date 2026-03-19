#!/bin/bash
# Consultancy experiment v3 — no grad clipping + advantage normalization
# v3: normalize_advantages_by_length=True, grad_clip=0.0 (no clipping)
#
# 3 seeds × G=8 = 3 runs (~$252)
#
# Launch from repo root: bash scripts/launch_consultancy_v3.sh
# Single run:            bash scripts/launch_consultancy_v3.sh 42
#
# Monitor: uv run python -m scripts.spot_check_rlvr logs/consultancy-experiment-v3/* --watch

set -euo pipefail

LOG_DIR="logs/consultancy-experiment-v3"
mkdir -p "$LOG_DIR"

COMMON=(
  model_name=Qwen/Qwen3.5-35B-A3B
  renderer_name=qwen3_5_disable_thinking
  dataset=consultancy_gpqa
  batch_size=64
  max_tokens=8192
  n_batches=50
  learning_rate=1e-5
  temperature=1.0
  eval_temperature=0.6
  eval_top_p=0.95
  loss_fn=ppo
  clip_ratio_upper=0.28
  advantage_scheme=maxrl
  kl_penalty_coef=0.0
  grad_clip_norm=0.0
  format_coef=0.10
  eos_coef=0.10
  eval_every=5
  save_every=5
  wandb_project=consultancy-gpqa
  behavior_if_log_dir_exists=resume
  normalize_advantages_by_length=True
)

IDX=0
run_one() {
  local SEED=$1
  local NAME="consult-v3-g8-s${SEED}"
  local DELAY=$((IDX * 90))
  echo "=== Launching ${NAME} (delay=${DELAY}s) ==="
  uv run --env-file .env python -m tinker_cookbook.recipes.consultancy.train \
    "${COMMON[@]}" \
    group_size=8 \
    seed="${SEED}" \
    log_path="${LOG_DIR}/${NAME}" \
    wandb_name="${NAME}" \
    initial_delay_s="${DELAY}" \
    > "${LOG_DIR}/${NAME}.stdout.log" 2>&1 &
  echo "  PID: $!  (log: ${LOG_DIR}/${NAME}.stdout.log)"
  IDX=$((IDX + 1))
}

# If args provided, run single seed
if [[ $# -eq 1 ]]; then
  run_one "$1"
  wait
  exit 0
fi

# Full matrix: 3 seeds
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
echo "  uv run python -m scripts.spot_check_rlvr ${LOG_DIR}/* --watch"
echo ""
echo "Kill all:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Tail a run:"
echo "  tail -f ${LOG_DIR}/consult-v3-g8-s42.stdout.log"
echo ""

# Wait for all runs; exit with failure if any run fails
FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=$((FAILED + 1))
done
if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/3 runs failed"
  exit 1
fi
echo "All 3 runs completed successfully."
