#!/bin/bash
# mets_analysis.sh
#SBATCH -J mets_analysis
#SBATCH --partition=l40s_nova
#SBATCH --gres=gpu:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --account=free

nvidia-smi
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

dataset_name="Dataset902_AutoPet"

image_dir="${nnUNet_results}/Dataset999_AutoPet/test/images"
label_dir="${nnUNet_results}/Dataset999_AutoPet/test/gt"
bone_dir="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/bone-metastasis/totalsegmentator/bone_segmentations_fold1"

for fold in "fold_0" "fold_1" "fold_2"; do
    pred_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/test_predictions"
    out_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}"

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
echo "Collating results across folds..."
python collate_results.py \
    --results-dir "${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/" \
    --folds fold_0 fold_1 fold_2 \
    --out-dir ./${dataset_name}_mets_results
