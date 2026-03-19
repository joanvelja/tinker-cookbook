#!/bin/bash
# GPQA RLVR experiment — 6 runs (2 group sizes x 3 seeds)
# Launch from repo root: bash scripts/launch_gpqa.sh
# Single run:            bash scripts/launch_gpqa.sh 8 42
#
# Matrix:
#   G={8, 16}, seed={42, 137, 7}
#   Model: Qwen3.5-35B-A3B, B=64, 50 steps, PPO w/ DAPO clip
#
# Monitor: uv run python -m scripts.spot_check_rlvr logs/gpqa-experiment/* --watch

set -euo pipefail

LOG_DIR="logs/gpqa-experiment-v2"
mkdir -p "$LOG_DIR"

# Grader: gpt-5-mini + default prompt (LLMGraderConfig defaults)
# No grader overrides needed — the builder defaults handle it.

COMMON=(
  model_name=Qwen/Qwen3.5-35B-A3B
  renderer_name=qwen3_5_disable_thinking
  dataset=gpqa_oe
  batch_size=64
  max_tokens=8192
  n_batches=50
  learning_rate=1e-5
  temperature=1.0
  top_p=0.95
  top_k=20
  eval_temperature=1.0
  eval_top_p=0.95
  eval_top_k=20
  loss_fn=ppo
  clip_ratio_upper=0.28
  advantage_scheme=maxrl
  kl_penalty_coef=0.0
  grad_clip_norm=0.3
  format_coef=0.10
  eos_coef=0.10
  eval_every=5
  save_every=5
  wandb_project=rlvr-gpqa
  behavior_if_log_dir_exists=resume
  normalize_advantages_by_length=True
  max_steps_off_policy=2
  grader_max_concurrent=1024
  num_substeps=2
)

IDX=0
run_one() {
  local G=$1 SEED=$2
  local NAME="gpqa-g${G}-s${SEED}"
  local DELAY=$((IDX * 90))  # stagger 90s apart
  echo "=== Launching ${NAME} (delay=${DELAY}s) ==="
  uv run --env-file .env python -m tinker_cookbook.recipes.rlvr.train \
    "${COMMON[@]}" \
    group_size="${G}" \
    seed="${SEED}" \
    log_path="${LOG_DIR}/${NAME}" \
    wandb_name="${NAME}" \
    initial_delay_s="${DELAY}" \
    > "${LOG_DIR}/${NAME}.stdout.log" 2>&1 &
  echo "  PID: $!  (log: ${LOG_DIR}/${NAME}.stdout.log)"
  IDX=$((IDX + 1))
}

# If args provided, run single configuration
if [[ $# -eq 2 ]]; then
  run_one "$1" "$2"
  wait
  exit 0
fi

# Full matrix: G={8,16} x seed={42,137,7}
PIDS=()
for G in 8 16; do
  for SEED in 42 137 7; do
    run_one "$G" "$SEED"
    PIDS+=($!)
  done
done

echo ""
echo "=== All 6 launched ==="
echo "  PIDs: ${PIDS[*]}"
echo ""
echo "Monitor:"
echo "  uv run python -m scripts.spot_check_rlvr ${LOG_DIR}/* --watch"
echo ""
echo "Kill all:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Tail a run:"
echo "  tail -f ${LOG_DIR}/gpqa-g8-s42.stdout.log"
echo ""

# Wait for all runs; exit with failure if any run fails
FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=$((FAILED + 1))
done
if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/6 runs failed"
  exit 1
fi
echo "All 6 runs completed successfully."
