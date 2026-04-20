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
#SBATCH --job-name=predict_tta_unc     # Descriptive job name
#SBATCH --account=bhattacharya-lab
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_indrani        # Partition (queue) name on your cluster
#SBATCH --time=1-12:00:00               # Walltime (days-HH:MM:SS) 
#SBATCH --nodelist=adanova01
#SBATCH --mem=60G                      # Memory request: 32 GB
#SBATCH --output=predict_tta_unc_%j.out  # Standard output log file
#SBATCH --error=predict_tta_unc_%j.err   # Standard error log file


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

fold_name="fold_0"
checkpoint_name="best"
MODEL_DIR="/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres"
INPUT_DIR="/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/train_link/"
OUTPUT_DIR="/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/train_tta_predicted/" 

python ../nnunetv2/inference/predict_TTA.py \
    --continue_prediction \
    -i $INPUT_DIR \
    -o $OUTPUT_DIR \
    -d 999 \
    -tr autoPET3_Trainer \
    -p nnUNetResEncUNetLPlansMultiTalent \
    -c 3d_fullres \
    -f 2 \
    -chk "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/checkpoint_${checkpoint_name}.pth" \
    --save_probabilities \
    --disable_tta 
# If you want to train multiple folds in one script, you could do something like:
# for FOLD in 0 1 2 3 4; do
#   nnUNetv2_train 888 2d $FOLD --npz
# done


echo "==============================="
