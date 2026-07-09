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

if [ -n "${GI_TRACT_DATASET_PATH:-}" ]; then
  GI_DATASET_PATH="$GI_TRACT_DATASET_PATH"
elif [ -f dataset_path.txt ]; then
  GI_DATASET_PATH="$(cat dataset_path.txt)"
else
  echo "ERROR: GI_TRACT_DATASET_PATH is not set and dataset_path.txt was not found in $(pwd)." >&2
  echo "Set GI_TRACT_DATASET_PATH before sbatch or create dataset_path.txt via: python3 download_dataset.py" >&2
  exit 1
fi

export GI_TRACT_DATASET_PATH="$GI_DATASET_PATH"

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

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: Neither 'python' nor 'python3' is available in PATH." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV_CMD=(uv)
elif "$PYTHON_BIN" -m uv --help >/dev/null 2>&1; then
  UV_CMD=("$PYTHON_BIN" -m uv)
else
  echo "ERROR: 'uv' is not available in PATH and '$PYTHON_BIN -m uv' is unavailable." >&2
  echo "Install uv or load the environment/module that provides it before submitting this job." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/prepare_segx_metadata.py \
  --dataset-root "$DATASET_ROOT" \
  --out-csv metadata/train.csv \
  --out-mask-dir metadata/masks

"${UV_CMD[@]}" run --project external/segx segx-build-folds \
  cfg_path=workdir/config/mri_segx.yaml \
  root_dir="$DATASET_ROOT" \
  splits_dir=workdir/cache/splits \
  seed=42

"${UV_CMD[@]}" run --project external/segx segx-train \
  config=workdir/config/mri_segx.yaml \
  data.datasets.root_dir="$DATASET_ROOT" \
  train.fold=0 \
  train.lr=1e-4

echo "Finished at: $(date)"
