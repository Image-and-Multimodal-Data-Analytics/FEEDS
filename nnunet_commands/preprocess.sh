#!/bin/bash

#SBATCH --job-name=preprocess_nnunet
#SBATCH --account=bhattacharya-lab
#SBATCH --partition=l40s_indrani
#SBATCH --nodelist=adanova01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=80G


# Check CUDA and GPU status
nvcc --version
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Please verify NVIDIA driver installation."
fi


# 2) Preprocess a new dataset 
nnUNet_def_n_proc=1 nnUNetv2_plan_and_preprocess -d 333 -c 3d_fullres -pl ResEncUNetPlanner -np 20 -npfp 20


