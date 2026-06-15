#!/bin/bash

#SBATCH --job-name=infer_test_DPSMA_%j        # gives infer_Test_0, infer_Test_1, etc
#SBATCH --output=logs/DPSMA_%j.out
#SBATCH --error=logs/DPSMA_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --array=0-3

# -----------------------------------------------------------------------------
# Print GPU and environment info for debugging
# -----------------------------------------------------------------------------
module load cuda/12.4
echo "==============================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==============================="
nvidia-smi
echo "==============================="

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

# Testing folds for 111 are fold 0, 5 and 8
# Testing folds for 999 are fold 2, 7 and 10 

# FOLDS=(fold_0 fold_5 fold_8)
FOLDS=(fold_2 fold_7 fold_10)
datasetid=999

fold_name=${FOLDS[$SLURM_ARRAY_TASK_ID]}   # picks fold based on job index

checkpoint_name="best"
dataset_name="Dataset${datasetid}_AutoPet"

echo "Running fold: $fold_name"

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
nnUNetv2_predict --c \
    -i "$nnUNet_results/../DPSMA/Dataset701_deep_psma/imagesTr" \
    -o "$nnUNet_results/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/DPSMA_predictions" \
    -d ${datasetid} \
    -tr autoPET3_Trainer \
    -p nnUNetResEncUNetLPlansMultiTalent \
    -c 3d_fullres \
    -chk "$nnUNet_results/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/checkpoint_${checkpoint_name}.pth" \
    --save_probabilities

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------
nnUNetv2_evaluate_folder \
    -djfile $nnUNet_results/$dataset_name/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/dataset.json \
    -pfile $nnUNet_results/$dataset_name/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/plans.json \
    $nnUNet_results/../DPSMA/Dataset701_deep_psma/labelsTr \
    $nnUNet_results/$dataset_name/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/DPSMA_predictions/
