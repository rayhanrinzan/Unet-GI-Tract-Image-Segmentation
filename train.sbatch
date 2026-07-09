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
mkdir -p workdir/cache/splits

export HOME=$BASE
export XDG_CACHE_HOME=$BASE/.cache
export GI_TRACT_DATASET_PATH="$(cat dataset_path.txt)"

export WANDB_DIR=$BASE/wandb
export WANDB_CACHE_DIR=$BASE/.cache/wandb
export WANDB_MODE=online

echo "Dataset path: $GI_TRACT_DATASET_PATH"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

DATASET_ROOT="$GI_TRACT_DATASET_PATH"
if [ "$(basename "$DATASET_ROOT")" != "dataset" ] && [ -d "$DATASET_ROOT/dataset" ]; then
  DATASET_ROOT="$DATASET_ROOT/dataset"
fi

python scripts/prepare_segx_metadata.py \
  --dataset-root "$GI_TRACT_DATASET_PATH" \
  --out-csv metadata/train.csv \
  --out-mask-dir metadata/masks

uv run --project external/segx segx-build-folds \
  cfg_path=workdir/config/mri_segx.yaml \
  root_dir="$DATASET_ROOT" \
  splits_dir=workdir/cache/splits \
  seed=42

uv run --project external/segx segx-train \
  config=workdir/config/mri_segx.yaml \
  data.datasets.root_dir="$DATASET_ROOT" \
  train.fold=0 \
  train.lr=1e-4

echo "Finished at: $(date)"
