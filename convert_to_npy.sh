#!/bin/bash

#SBATCH --job-name=convert_npy
#SBATCH --account=bhattacharya-lab
#SBATCH --partition=l40s_indrani
#SBATCH --gres=gpu:1
#SBATCH --output=convert_npy_%j.out
#SBATCH --error=convert_npy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=40G


# Check CUDA and GPU status
nvcc --version
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Please verify NVIDIA driver installation."
fi


# 2) Preprocess a new dataset 
python convert_to_npy.py

