#!/bin/bash
# mets_analysis.sh
#SBATCH -J run_both_analysis
#SBATCH --partition=gpuq
#SBATCH --account=free
#SBATCH -o logs/both%j.out
#SBATCH -e logs/both%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24            
#SBATCH --time=10:30:00
#SBATCH --mem=64G

nvidia-smi
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Define all folds in an array
# folds=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4" "fold_5" "fold_6" "fold_7" "fold_8" "fold_9" "fold_10" "fold_11")

# if no $SLURM_ARRAY_TASK_ID; then
#     echo "Error: SLURM_ARRAY_TASK_ID is not set. Please run this script as part of a SLURM array job."
#     exit 1
# fi

# # Pick the fold corresponding to the array task ID
# fold=${folds[$SLURM_ARRAY_TASK_ID]}

# image_dir="${nnUNet_results}/test/images"
# label_dir="${nnUNet_results}/test/gt"
# version="test_predictions"


# image_dir="${nnUNet_results}/Dataset999_AutoPet/val/images"

dataset_name="Dataset904_AutoPet"
version="DHMC_predictions"
label_dir="/dartfs/rc/lab/B/BhattacharyaI/DHMC_data/gt"
image_dir="/dartfs/rc/lab/B/BhattacharyaI/DHMC_data/images_with_annotation"
bone_dir="/dartfs/rc/lab/B/BhattacharyaI/DHMC_data/images_with_annotation/ts_segmentations"

echo "Processing fold: $fold"
for fold in "fold_0" "fold_1" "fold_2"; do
    echo "Processing fold: $fold"
    pred_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/${version}"
    out_dir="${pred_dir}/lesion_metrics.csv"
    python get_fp_fn_dice.py \
        --pred_dir $pred_dir \
        --gt_dir $label_dir \
        --save_dir $out_dir \
        --num_workers 4

    echo "Calculating mets metrics for fold: $fold"
    n_out_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}"
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

done
