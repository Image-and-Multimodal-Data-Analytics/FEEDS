#!/bin/bash
# mets_analysis.sh
#SBATCH -J get_fp_fn
#SBATCH --partition=cpu
#SBATCH -o logs/lesion_metric_calc_%j.out
#SBATCH -e logs/lesion_metric_calc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24            
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --array=0-10   # adjust range to match your folds (0-11 = fold_0 to fold_11)

nvidia-smi
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Define all folds in an array
folds=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4" "fold_5" "fold_6" "fold_7" "fold_8" "fold_9" "fold_10" "fold_11")

# if no $SLURM_ARRAY_TASK_ID; then
#     echo "Error: SLURM_ARRAY_TASK_ID is not set. Please run this script as part of a SLURM array job."
#     exit 1
# fi

# # Pick the fold corresponding to the array task ID
fold=${folds[$SLURM_ARRAY_TASK_ID]}

# image_dir="${nnUNet_results}/test/images"
# label_dir="${nnUNet_results}/test/gt"

image_dir="${nnUNet_results}/val/images"
label_dir="${nnUNet_results}/val/gt"

version="validation_279"
dataset_name="Dataset111_AutoPet"

echo "Processing fold: $fold"

pred_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/${version}"
out_dir="${pred_dir}/lesion_metrics.csv"
python get_fp_fn_dice.py \
    --pred_dir $pred_dir \
    --gt_dir $label_dir \
    --save_dir $out_dir \
    --num_workers 4

