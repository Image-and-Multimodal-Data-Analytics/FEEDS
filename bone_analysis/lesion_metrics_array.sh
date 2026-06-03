#!/bin/bash
# mets_analysis.sh
#SBATCH -J mets_analysis_%A_%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=0-2    # adjust range to match your folds (0-2 = fold_2 to fold_11)

nvidia-smi
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

dataset_name="Dataset999_AutoPet"

# Define all folds in an array
folds=("fold_2" "fold_9" "fold_11")

# Pick the fold corresponding to the array task ID
fold=${folds[$SLURM_ARRAY_TASK_ID]}

image_dir="${nnUNet_results}/test/images"
label_dir="${nnUNet_results}/test/gt"
bone_dir="${nnUNet_results}/ts_segmentations"

pred_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/test_predictions"
out_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}"

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running mets analysis for fold: $fold"
echo "Image dir: $image_dir"
echo "Label dir: $label_dir"
echo "Bone dir: $bone_dir"
echo "Prediction dir: $pred_dir"
echo "Output dir: $out_dir"

python main.py \
    --mode mets \
    -f  "$fold" \
    -i  "$image_dir" \
    -ld "$label_dir" \
    -b  "$bone_dir" \
    -p  "$pred_dir" \
    -o  "$out_dir" \
    -j  8 \
    --skip-existing