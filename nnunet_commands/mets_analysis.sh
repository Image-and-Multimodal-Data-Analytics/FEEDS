#!/bin/bash
#SBATCH -J 012
#SBATCH --partition=l40s_nova
#SBATCH --gres=gpu:1
#SBATCH -o filename_%j.out          # File to save standard output
#SBATCH -e filename_%j.err          # File to save standard error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=60
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --account=free
# Print GPU status and hostname (these will be logged in the output file)
nvidia-smi
hostname

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# loop through the folds

dataset_name="Dataset902_AutoPet"

image_dir="${nnUNet_results}/Dataset999_AutoPet/test/images"
label_dir="${nnUNet_results}/Dataset999_AutoPet/test/gt"
bone_dir="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/bone-metastasis/totalsegmentator/bone_segmentations_fold1"

for fold in "fold_0" "fold_1" "fold_2"
do
    pred_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/$fold/test_predictions"
    out_dir="${nnUNet_results}/${dataset_name}/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/"
    python main_mets.py -f $fold -i $image_dir -p $pred_dir -b $bone_dir -ld $label_dir -o $out_dir
done



