#!/bin/bash
# GPQA RLVR — Qwen3-30B-A3B-Instruct-2507 (no-think natively)
# 3 versions × 3 seeds × G=8 = 9 runs total
#
# v1: token_sum baseline (no advantage norm, grad_clip=0.3)
# v2: normalize_advantages_by_length=True, grad_clip=0.3
# v3: normalize_advantages_by_length=True, grad_clip=0.0
#
# Launch: bash scripts/launch_gpqa_q3instruct.sh
# Single: bash scripts/launch_gpqa_q3instruct.sh v1 42
#
# Monitor: uv run python -m scripts.spot_check_rlvr logs/gpqa-q3instruct/* --watch

set -euo pipefail

LOG_DIR="logs/gpqa-q3instruct"
mkdir -p "$LOG_DIR"

COMMON=(
  model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
  renderer_name=qwen3_instruct
  dataset=gpqa_oe
  batch_size=64
  group_size=8
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
  format_coef=0.10
  eos_coef=0.10
  eval_every=5
  save_every=5
  wandb_project=rlvr-gpqa-q3instruct
  behavior_if_log_dir_exists=resume
  max_steps_off_policy=2
  grader_max_concurrent=1024
  num_substeps=2
)

IDX=0
run_one() {
  local VERSION=$1 SEED=$2
  local NAME="q3-${VERSION}-g8-s${SEED}"

  # Version-specific overrides
  local EXTRA=()
  case "$VERSION" in
    v1) EXTRA=(normalize_advantages_by_length=False grad_clip_norm=0.3) ;;
    v2) EXTRA=(normalize_advantages_by_length=True  grad_clip_norm=0.3) ;;
    v3) EXTRA=(normalize_advantages_by_length=True  grad_clip_norm=0.0) ;;
    *)  echo "Unknown version: $VERSION"; exit 1 ;;
  esac

  local DELAY=$((IDX * 90))
  echo "=== Launching ${NAME} (delay=${DELAY}s) ==="
  uv run --env-file .env python -m tinker_cookbook.recipes.rlvr.train \
    "${COMMON[@]}" \
    "${EXTRA[@]}" \
    seed="${SEED}" \
    log_path="${LOG_DIR}/${NAME}" \
    wandb_name="${NAME}" \
    initial_delay_s="${DELAY}" \
    > "${LOG_DIR}/${NAME}.stdout.log" 2>&1 &
  echo "  PID: $!  (log: ${LOG_DIR}/${NAME}.stdout.log)"
  IDX=$((IDX + 1))
}

# Single run mode: bash launch_gpqa_q3instruct.sh v2 42
if [[ $# -eq 2 ]]; then
  run_one "$1" "$2"
  wait
  exit 0
fi

# Full matrix: 3 versions × 3 seeds = 9 runs
PIDS=()
for VERSION in v1 v2 v3; do
  for SEED in 42 137 7; do
    run_one "$VERSION" "$SEED"
    PIDS+=($!)
  done
done

echo ""
echo "=== All 9 launched ==="
echo "  PIDs: ${PIDS[*]}"
echo ""
echo "Monitor:"
echo "  uv run python -m scripts.spot_check_rlvr ${LOG_DIR}/* --watch"
echo ""
echo "Kill all:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Tail a run:"
echo "  tail -f ${LOG_DIR}/v1-g8-s42.stdout.log"
echo ""

FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=$((FAILED + 1))
done
if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/9 runs failed"
  exit 1
fi
echo "All 9 runs completed successfully."
