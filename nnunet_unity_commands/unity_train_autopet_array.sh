#!/bin/bash

#SBATCH --job-name=330_fold
#SBATCH --output=autopet3_train_%A_%a.out
#SBATCH --error=autopet3_train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --array=1-4  # folds 0,1,2,3,4 — adjust range as needed

# Clean caches
rm -rf /scratch/f007g3j/torchinductor_f007g3j/

# Safe env
export nnUNet_compile=False
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
export CUDA_LAUNCH_BLOCKING=1     # keep while debugging

# The fold is taken from the array task ID
FOLD=${SLURM_ARRAY_TASK_ID}

echo "Using Python at: $(which python)"
echo "Running fold: ${FOLD}"

# Training with pretrained weights + uncertainty maps
nnUNetv2_train 330 3d_fullres ${FOLD} \
    -tr autoPET3_Trainer \
    -p nnUNetResEncUNetLPlansMultiTalent \
    -pretrained_weights ~/pretrained_final_checkpoint.pth \
    --npz

echo "Training fold ${FOLD} completed at: $(date)"