#!/bin/bash
#SBATCH --job-name=CLIP_Fine_Tuning
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-20
#SBATCH --output=logs/job_%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"

# Sweep range: 3..12 inclusive (10 values each -> 100 combos)
MIN=3
N=10

N_TEXT=8   #$(( MIN + SLURM_ARRAY_TASK_ID / N ))
N_VISION=3 #$(( MIN + SLURM_ARRAY_TASK_ID % N ))

# One head config per entry: "name|extra args"
CONFIGS=(
  "linear|--fusion concat3 --hidden 0"
  "mlp256_d1|--fusion concat3 --hidden 256 --dropout 0.1"
  "mlp256_d3|--fusion concat3 --hidden 256 --dropout 0.3"
  "mlp512_d3|--fusion concat3 --hidden 512 --dropout 0.3"
  "mlp512_d5|--fusion concat3 --hidden 512 --dropout 0.5"
  "mlp1024_d3|--fusion concat3 --hidden 1024 --dropout 0.3"
  "concat2_mlp512|--fusion concat2 --hidden 512 --dropout 0.3"
)
SEEDS=(1 2 3)

N_SEEDS=${#SEEDS[@]}
CFG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

ENTRY="${CONFIGS[$CFG_IDX]}"
NAME="${ENTRY%%|*}"
ARGS="${ENTRY#*|}"

echo "Task ${SLURM_ARRAY_TASK_ID}: head=${NAME} seed=${SEED} args=${ARGS}"

.venv/bin/python CLIP_Fine_Tuning_1.py \
    --n-vision 3 --n-text 8 --epochs 3 --select-metric f1_macro \
    --seed "$SEED" --out "runs/head_${NAME}_s${SEED}" \
    $ARGS "$@"