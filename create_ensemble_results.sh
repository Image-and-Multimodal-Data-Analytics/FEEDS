#!/bin/bash

#SBATCH --job-name=train_ensemble
#SBATCH --account=bhattacharya-lab-share
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=create_ensemble_results_%j.out
#SBATCH --error=create_ensemble_results_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=10G


# Check CUDA and GPU status
nvcc --version
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Please verify NVIDIA driver installation."
fi


# 2) Preprocess a new dataset 
python collate_uncertainty_maps.py

