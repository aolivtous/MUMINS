#!/bin/bash
#------------------------------------------------------------------
# MUMINS - FSDP training job
#
# Before running:
#   1. Edit the paths in the "EDIT THESE" block below for your machine.
#   2. Adjust the #SBATCH lines for your cluster (partition, GPU type,
#      walltime, etc.). --ntasks and --gres GPU count should match
#      (one task per GPU for FSDP).
#   3. sbatch run_train_fsdp.sh
#
# Do not commit this file back to the repo with your real paths or
# API keys filled in - keep your edited copy local, or track your
# changes in a separate branch/fork.
#------------------------------------------------------------------
#SBATCH -J mumins_train_fsdp
#SBATCH -o jobs_train/job_%j.o          # make sure ./logs exists, or change this
#SBATCH -e jobs_train/job_%j.e
#SBATCH --partition=gpu           # EDIT: your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=2                # one task per GPU (FSDP)
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:ampere:2       # EDIT: GPU type/count for your cluster - match --ntasks

set -euo pipefail

#================== EDIT THESE for your machine ==================#
PROJECT_ROOT="/path/to/MUMINS"
CONDA_MODULE="conda"                            # module name for `module load`, if used
CONDA_ENV_PATH="/path/to/conda/envs/mumins"
OASIS_DATA_ROOT="/path/to/OASIS-3_preprocessed_turboprep_1p5mm/our_B_filtered.csv"
NGP_DATA_ROOT="/path/to/lung_nodule_growth"
RESULTS_ROOT="/path/to/MUMINS/outputs"
DATASET="OASIS"
CH=2
LEARN_VAR=True
TIMESTEPS=300
DIFF=True
SIZE=128
BATCH_SIZE=1              # per-GPU batch size under FSDP (see note below)
USE_CHECKPOINT=True       # gradient checkpointing, helps at higher resolutions
#===================================================================#

#------------------ Environment ------------------#
module load "${CONDA_MODULE}"
source activate "${CONDA_ENV_PATH}"

echo "Running on host: $(hostname)"
echo "Available GPUs: ${CUDA_VISIBLE_DEVICES:-none}"

#------------------ Distributed setup ------------------#
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$DATASET" in
    OASIS) DATA_ROOT="$OASIS_DATA_ROOT" ;;
    NGP)   DATA_ROOT="$NGP_DATA_ROOT" ;;
    *)     echo "Error: Unknown dataset '$DATASET'" >&2; exit 1 ;;
esac

# NOTE: under FSDP, batch_size is PER GPU (a DistributedSampler splits
# the dataset across ranks). Effective global batch size = BATCH_SIZE x
# number of GPUs (--ntasks above). 

#------------------ Weights & Biases ------------------#

export WANDB_DIR="${WANDB_DIR:-${RESULTS_ROOT}/wandb_logs}"
mkdir -p "$WANDB_DIR"
# SECURITY: do not hardcode your API key here. Run `wandb login` once on this
# machine, or export WANDB_API_KEY from a private file/secret manager before
# launching this script.


#------------------ Output path ------------------#
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="${RESULTS_ROOT}/checkpoints/MUMINS_${DATASET}_${CH}ch_var${LEARN_VAR}_${TIMESTEPS}s_diff${DIFF}_size${SIZE}_fsdp/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

#------------------ Run ------------------#
srun --ntasks=2 --gpus-per-task=1 python -B "${PROJECT_ROOT}/train/train.py" \
    model=ddpm \
    dataset="$DATASET" \
    dataset.root_dir="$DATA_ROOT" \
    dataset.mode=train \
    dataset.diff=$DIFF \
    model.diffusion_img_size=$SIZE \
    model.diffusion_depth_size=$SIZE \
    model.diffusion_num_channels=$CH \
    model.batch_size=$BATCH_SIZE \
    model.results_folder="$RESULTS_DIR" \
    model.load_milestone=False \
    model.learned_variance=$LEARN_VAR \
    model.save_and_sample_every=50 \
    model.train_num_steps=150001 \
    model.timesteps=$TIMESTEPS \
    model.use_checkpoint=$USE_CHECKPOINT \
    model.num_workers=20