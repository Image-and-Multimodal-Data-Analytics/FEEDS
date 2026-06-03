#!/bin/bash
# mets_analysis.sh
#SBATCH -J mets_analysis_9_8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

nvidia-smi
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


image_dir="${nnUNet_results}/test/images"
label_dir="${nnUNet_results}/test/gt"
bone_dir="${nnUNet_results}/ts_segmentations"

dataset_name="Dataset999_AutoPet"
for fold in "fold_7"; do
    pred_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/test_predictions"
    out_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}"

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
done

