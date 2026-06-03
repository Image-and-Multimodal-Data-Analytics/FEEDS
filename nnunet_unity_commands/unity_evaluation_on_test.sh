#!/bin/bash

# -----------------------------------------------------------------------------
# Basic Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --job-name=evaluation_test
#SBATCH --output=autopet3_train_%j.out
#SBATCH --error=autopet3_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mem=40G

# -----------------------------------------------------------------------------
# Print GPU and environment info for debugging
# -----------------------------------------------------------------------------
# module load cuda/12.4
echo "==============================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==============================="
nvidia-smi
echo "==============================="

# -----------------------------------------------------------------------------
# (1) Load modules or activate environmentsq
# -----------------------------------------------------------------------------
# Example commands (customize for your cluster):
# module load cuda/11.7
# module load anaconda/2023
# source activate nnunet_env

# for fold in fold_10_ens_0 fold_10_ens_1 fold_10_ens_2 fold_10_ens_3 fold_10_ens_4; do

input_dataset="Dataset999_AutoPet"
validation="test_predictions"


for fold in fold_0 fold_3; do
  echo "Evaluating fold: $fold"
  nnUNetv2_evaluate_folder \
    -djfile $nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/dataset.json \
    -pfile $nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/plans.json \
    $nnUNet_results/test/gt \
    $nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/${validation}/ 
done
