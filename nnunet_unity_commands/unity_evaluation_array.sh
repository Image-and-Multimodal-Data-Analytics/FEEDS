#!/bin/bash

# -----------------------------------------------------------------------------
# Basic Slurm directives
# -----------------------------------------------------------------------------
#SBATCH --job-name=evaluation_val
#SBATCH --output=autopet3_eval_%A_%a.out
#SBATCH --error=autopet3_eval_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=0:15:00
#SBATCH --mem=40G
#SBATCH --array=9       # 9 folds (fold_0 to fold_8)

# -----------------------------------------------------------------------------
# Print GPU and environment info for debugging
# -----------------------------------------------------------------------------
echo "==============================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "==============================="
nvidia-smi
echo "==============================="

# -----------------------------------------------------------------------------
# Map SLURM array task ID to fold name
# -----------------------------------------------------------------------------
FOLDS=(fold_0 fold_1 fold_2 fold_3 fold_4 fold_5 fold_6 fold_7 fold_8 fold_9 fold_10 fold_11)
fold=${FOLDS[$SLURM_ARRAY_TASK_ID]}

input_dataset="Dataset999_AutoPet"
validation="validation_279"

# -----------------------------------------------------------------------------
# Run evaluation for the assigned fold
# -----------------------------------------------------------------------------
echo "Evaluating fold: $fold"

nnUNetv2_evaluate_folder \
    -djfile $nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/dataset.json \
    -pfile $nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/plans.json \
    $nnUNet_results/val/gt \
    $nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/${validation}/
