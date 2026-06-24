#!/bin/bash -l

#SBATCH --job-name=unet_gi_mri
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

mkdir -p logs outputs

module purge
module load miniconda3/22.11.1-ctkwnpe
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mri

export GI_TRACT_DATASET_PATH="$(cat dataset_path.txt)"

echo "Dataset path: $GI_TRACT_DATASET_PATH"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

python train.py \
--epochs 10 \
--batch-size 8 \
--num-workers 1 \
--output-dir outputs

echo "Finished at: $(date)"
