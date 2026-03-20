#!/bin/bash
# GPQA RLVR — Q3-30B-A3B-Instruct, G=16, 3 seeds
#
# Config: B=64, G=16, 32 steps (4 epochs over 491 train problems)
# normalize_advantages_by_length=True, grad_clip=0.3
# async pipelining (max_steps_off_policy=2), num_substeps=2
#
# Cost estimate: ~$81/seed, ~$243 total (lower bound)
#
# Launch: bash scripts/launch_rlvr_q3_g16.sh
# Single: bash scripts/launch_rlvr_q3_g16.sh 42
#
# Monitor: uv run python -m scripts.spot_check_rlvr logs/rlvr-q3-g16/* --watch

set -euo pipefail

LOG_DIR="logs/rlvr-q3-g16"
mkdir -p "$LOG_DIR"

COMMON=(
  model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
  renderer_name=qwen3_instruct
  dataset=gpqa_oe
  batch_size=64
  group_size=16
  max_tokens=8192
  n_batches=32
  learning_rate=1e-5
  temperature=1.0
  top_p=0.8
  top_k=20
  eval_temperature=0.7
  eval_top_p=0.8
  eval_top_k=20
  loss_fn=ppo
  clip_ratio_upper=0.28
  advantage_scheme=maxrl
  kl_penalty_coef=0.0
  format_coef=0.10
  eos_coef=0.10
  normalize_advantages_by_length=True
  grad_clip_norm=0.0
  eval_every=4
  save_every=4
  wandb_project=gpqa-oe-main-fig
  behavior_if_log_dir_exists=resume
  max_steps_off_policy=2
  num_substeps=2
  grader_max_concurrent=1024
)

IDX=0
run_one() {
  local SEED=$1
  local NAME="rlvr-q3-g16-s${SEED}"
  local DELAY=$((IDX * 90))
  echo "=== Launching ${NAME} (delay=${DELAY}s) ==="
  uv run --env-file .env python -m tinker_cookbook.recipes.rlvr.train \
    "${COMMON[@]}" \
    seed="${SEED}" \
    log_path="${LOG_DIR}/${NAME}" \
    wandb_name="${NAME}" \
    initial_delay_s="${DELAY}" \
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
echo "  uv run python -m scripts.spot_check_rlvr ${LOG_DIR}/* --watch"
echo ""
echo "Kill all:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Tail a run:"
echo "  tail -f ${LOG_DIR}/rlvr-q3-g16-s42.stdout.log"
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
