#!/bin/bash
#SBATCH --job-name=debate-protocol
#SBATCH --output=logs/protocol-experiment/slurm_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal

# Protocol comparison experiment — runs on MATS cluster
# Submit: sbatch scripts/slurm_protocol_experiment.sh
# Monitor: squeue -u $USER
# Logs: logs/protocol-experiment/slurm_<jobid>.log

cd "$HOME/tinker-cookbook" || exit 1

# Ensure deps are installed
uv sync --quiet

bash scripts/launch_protocol_experiment.sh

# Wait for all background jobs
wait
echo "All runs completed at $(date)"
