#!/bin/bash -l

#SBATCH --job-name=unet_gi_mri
#SBATCH --partition=preempt_gpu
#SBATCH --account=imedslab
#SBATCH --qos=low
#SBATCH --chdir=/midtier/imedslab/rar4037/gi_mri_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

BASE=/midtier/imedslab/rar4037

mkdir -p outputs
mkdir -p $BASE/wandb
mkdir -p $BASE/.cache/wandb

module purge
module load miniconda3/22.11.1-ctkwnpe

echo "1"

export HOME=$BASE
export XDG_CACHE_HOME=$BASE/.cache
export GI_TRACT_DATASET_PATH="$(cat dataset_path.txt)"

echo "Dataset path: $GI_TRACT_DATASET_PATH"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

python train.py \
--epochs 1 \
--batch-size 2 \
--num-workers 0 \
--output-dir outputs \
--no-augment

echo "Finished at: $(date)"
