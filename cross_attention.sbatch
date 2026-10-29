#!/bin/bash
#SBATCH --job-name=cross-attn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi

.venv/bin/python cross_attention.py "$@"
