#!/bin/bash
#------------------------------------------------------------------
# MUMINS - Inference / test job
#
# Before running:
#   1. Edit the paths in the "EDIT THESE" block below for your machine.
#   2. Adjust the #SBATCH lines for your cluster (partition, GPU type,
#      walltime, etc.).
#   3. sbatch run_inference.slurm.sh [DATASET] [MODEL_DIR]
#
# Do not commit this file back to the repo with your real paths or
# API keys filled in - keep your edited copy local, or track your
# changes in a separate branch/fork.
#------------------------------------------------------------------
#SBATCH -J mumins_inference
#SBATCH -o jobs_test/job_%j.o         
#SBATCH -e jobs_test/job_%j.e
#SBATCH --partition=gpu           # EDIT: your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:ampere:1      # EDIT: GPU type/count for your cluster

set -euo pipefail

#================== EDIT THESE ==================#
PROJECT_ROOT="/path/to/MUMINS"
CONDA_MODULE="conda" # module name for `module load`, if used
CONDA_ENV_PATH="/path/to/conda/envs/mumins"
OASIS_DATA_ROOT="/path/to/OASIS-3_preprocessed_turboprep_1p5mm/our_B_filtered.csv"
NGP_DATA_ROOT="/path/to/lung_nodule_growth"
RESULTS_ROOT="/path/to/MUMINS/outputs"
# Example for an OASIS (brain MRI) checkpoint:
# MODEL_DIR="/path/to/MUMINS/outputs/checkpoints/MUMINS_OASIS_2ch_varTrue_300s_diffTrue_size128/<timestamp>"
# MODEL_NUM=2705
MODEL_DIR="/path/to/MUMINS/outputs/checkpoints/MUMINS_NGP_2ch_varTrue_300s_diffTrue_size64/<timestamp>"
MODEL_NUM=3000
DATASET="NGP"
SIZE=64
TRAINED_WITH_FSDP=False
LEARN_VAR=True
TIMESTEPS=300
DIFF=True
SAMPLER=ddim          # 'ddpm' or 'ddim'
SKIP_INTERVAL=5        # only used for ddim
VAR_START=0.05          # fraction of timesteps before variance propagation starts
CH=2
DIFF_BLUR_MODE="zero"  # "zero", "gt", "mean", "median"
MODE="test"  # "train" or "test" or "valid"
ETA=1
#===================================================================#

#------------------ Environment ------------------#
module load "${CONDA_MODULE}"
source activate "${CONDA_ENV_PATH}"

echo "Running on host: $(hostname)"
echo "Available GPUs: ${CUDA_VISIBLE_DEVICES:-none}"

case "$DATASET" in
    OASIS) DATA_ROOT="$OASIS_DATA_ROOT" ;;
    NGP)   DATA_ROOT="$NGP_DATA_ROOT" ;;
    *)     echo "Error: Unknown dataset '$DATASET'" >&2; exit 1 ;;
esac

if [ -z "$MODEL_DIR" ]; then
    echo "Error: no model checkpoint directory given." >&2
    echo "Usage: sbatch run_inference.slurm.sh <DATASET> <MODEL_DIR>" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model directory does not exist: $MODEL_DIR" >&2
    exit 1
fi

#------------------ Output path ------------------#
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR="${RESULTS_ROOT}/inference_visualization/${DATASET}_${MODE}split_${DIFF_BLUR_MODE}DiffBlur_var${LEARN_VAR}_size${SIZE}_${TIMESTEPS}s_${SAMPLER}_diff${DIFF}_eta${ETA}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

#------------------ Run ------------------#
python -B "${PROJECT_ROOT}/test/inference.py" \
    model_path="$MODEL_DIR" \
    model_num="$MODEL_NUM" \
    dataset="$DATASET" \
    mode="$MODE" \
    diffusion_img_size=$SIZE \
    diffusion_depth_size=$SIZE \
    diffusion_num_channels=$CH \
    learned_variance=$LEARN_VAR \
    diff=$DIFF \
    timesteps=$TIMESTEPS \
    use_metadata=True \
    sampler=$SAMPLER \
    skip_interval=$SKIP_INTERVAL \
    variance_start=$VAR_START \
    root_dir="$DATA_ROOT" \
    trained_with_fsdp=$TRAINED_WITH_FSDP \
    out_dir="$OUT_DIR" \
    diff_blur_mode=$DIFF_BLUR_MODE \
    base_seed=42 \
    num_seeds=1 \
    eta=$ETA \
