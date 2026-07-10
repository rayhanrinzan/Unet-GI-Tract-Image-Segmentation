#!/bin/bash -l

#SBATCH --job-name=unet_gi_mri
#SBATCH --partition=preempt_gpu
#SBATCH --account=imedslab
#SBATCH --qos=low
#SBATCH --chdir=/midtier/imedslab/rar4037/gi_mri_project
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

BASE=/midtier/imedslab/rar4037

mkdir -p outputs
mkdir -p $BASE/.cache
mkdir -p $BASE/wandb
mkdir -p $BASE/.cache/wandb

export HOME=$BASE
export XDG_CACHE_HOME=$BASE/.cache
export GI_TRACT_DATASET_PATH="$(cat dataset_path.txt)"
unset UWMGI_MASK_DATASET_PATH

export WANDB_DIR=$BASE/wandb
export WANDB_CACHE_DIR=$BASE/.cache/wandb
export WANDB_MODE=online

echo "Dataset path: $GI_TRACT_DATASET_PATH"
echo "Precomputed mask dataset: disabled (using contour RLE masks)"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

echo "About to start main.py with W&B..."
echo "WANDB_MODE=${WANDB_MODE:-not set}"
echo "WANDB_DIR=${WANDB_DIR:-not set}"
echo "WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-not set}"

$BASE/conda_envs/mri/bin/python -u main.py \
--epochs 15 \
--batch-size 32 \
--learning-rate 1e-4 \
--num-workers 2 \
--output-dir outputs \
--use-wandb \
--wandb-project "MRI Scans 5" \
--wandb-run-name "unet_slurm_${SLURM_JOB_ID}"

echo "Finished at: $(date)"
