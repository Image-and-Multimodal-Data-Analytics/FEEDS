#!/bin/bash

#SBATCH --job-name=convert_npy
#SBATCH --account=bhattacharya-lab
#SBATCH --partition=l40s_indrani
#SBATCH --gres=gpu:1
#SBATCH --output=convert_npy_%j.out
#SBATCH --error=convert_npy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-07:00:00
#SBATCH --mem=40G


# Check CUDA and GPU status
nvcc --version
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Please verify NVIDIA driver installation."
fi


dataset="Dataset111_AutoPet"
fold="fold_5"
# 2) Preprocess a new dataset 
python convert_to_npy.py --nifti_labels_dir $nnUNet_results/$dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/$fold/train_predictions_60/ \
    --pkl_dir $nnUNet_preprocessed/$dataset/nnUNetPlans_3d_fullres/ \
    --output_dir $nnUNet_results/$dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/$fold/train_predictions_60/converted_segs

