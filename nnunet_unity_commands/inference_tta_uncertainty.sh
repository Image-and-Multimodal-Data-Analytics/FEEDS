#!/bin/bash 

# =============================================================================
# TTA Uncertainty Inference - nnU-Net
# =============================================================================
# This script runs inference with explicit Test Time Augmentation (TTA),
# saves individual TTA probability volumes, and computes entropy-based
# uncertainty maps (aleatoric, epistemic, total).
#
# Output structure:
#   - predictions/         : averaged segmentations
#   - tta_probabilities/   : per-case stacked TTA probabilities
#   - uncertainty_maps/    : globally normalized uncertainty maps
#   - uncertainty_maps_raw/: raw (unnormalized) uncertainty maps
# =============================================================================

# =============================================================================
# Basic Slurm directives
# =============================================================================
#SBATCH --job-name=tta
#SBATCH --output=tta%j.out
#SBATCH --error=tta%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=40G


# =============================================================================
# Print GPU and environment info for debugging
# =============================================================================
module load cuda/12.4
echo "==============================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==============================="
nvidia-smi
echo "==============================="

# =============================================================================
# Configuration - Edit these paths for your inference run
# =============================================================================

# Path to the trained model folder (contains fold_X subdirectories)
# Which folds to use (space-separated, or 'all')


# Sliding window step size (0-1; smaller = more overlap, slower but more accurate)

# Image filename to look for in each case folder
IMAGE_NAME=".nii.gz"

# =============================================================================
# Run TTA Uncertainty Inference
# =============================================================================

#!/bin/bash

fold_name="fold_8"
checkpoint_name="best"
MODEL_DIR="$nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres"
INPUT_DIR="$nnUNet_results/train_30_pl"
OUTPUT_DIR="$nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/train_tta/" 

python ../nnunetv2/inference/predict_TTA.py \
    --continue_prediction \
    -i $INPUT_DIR \
    -o $OUTPUT_DIR \
    -d 111 \
    -tr autoPET3_Trainer \
    -p nnUNetResEncUNetLPlansMultiTalent \
    -c 3d_fullres \
    -f 0 \
    -chk "$nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/checkpoint_${checkpoint_name}.pth" \
    --save_probabilities \
    --disable_tta 
# If you want to train multiple folds in one script, you could do something like:
# for FOLD in 0 1 2 3 4; do
#   nnUNetv2_train 888 2d $FOLD --npz
# done


echo "==============================="
