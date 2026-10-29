#!/bin/bash
#SBATCH --job-name=CLIP_LoRA_fusion_classifier
#SBATCH --gres=shard:nvidia_l40s:6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/CLIP_LoRA_fusion_classifier_job_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"

.venv/bin/python CLIP_LoRA_fusion_classifier.py "$@"
