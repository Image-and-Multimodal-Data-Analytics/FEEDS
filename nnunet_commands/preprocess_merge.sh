#!/bin/bash

#SBATCH --job-name=preprocess_nnUNet
#SBATCH --account=bhattacharya-lab-share
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=40:00:00
#SBATCH --mem=80G


# Check CUDA and GPU status
nvcc --version
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Please verify NVIDIA driver installation."
fi


# 2) Preprocess organ dataset (200) in 3D
# nnUNetv2_plan_and_preprocess -d 200 -c 3d_fullres -pl ResEncUNetPlanner -np 20 -npfp 20

# 3) Merge the two 3D-preprocessed datasets
# nnUNetv2_merge_lesion_and_organ_dataset -l 888 -o 200
# nnUNetv2_merge_lesion_and_organ_dataset -l 777 -o 200
# nnUNetv2_merge_lesion_and_organ_dataset -l 333 -o 200

python ../nnunetv2/preprocessing/organ_extraction/combine_lesion_and_organs.py -l 333 -o 200