#!/bin/bash
#SBATCH --array=7-9
#SBATCH --job-name=infer_test_%j        # gives infer_Test_0, infer_Test_1, etc
#SBATCH --output=logs/infer_test_%j.out
#SBATCH --error=logs/infer_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=40G
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
FOLDS=(fold_0 fold_1 fold_2 fold_3 fold_4 fold_5 fold_6 fold_7 fold_8 fold_9)
fold_name=${FOLDS[$SLURM_ARRAY_TASK_ID]}   # picks fold based on job index

checkpoint_name="best"
datasetid=999
dataset_name="Dataset${datasetid}_AutoPet"

echo "Running fold: $fold_name"

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
nnUNetv2_predict --c \
    -i "$nnUNet_results/test/images" \
    -o "$nnUNet_results/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/test_predictions" \
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
    $nnUNet_results/test/gt \
    $nnUNet_results/$dataset_name/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/test_predictions/
