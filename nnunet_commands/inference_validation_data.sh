#!/bin/bash 

# -----------------------------------------------------------------------------
# Basic Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --job-name=f6_val   # Descriptive job name
#SBATCH --account=bhattacharya-lab  
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_indrani         # Partition (queue) name on your cluster
#SBATCH --time=1-23:00:00              # Walltime (HH:MM:SS) 
#SBATCH --mem=32G                    # Memory request: 16 GB

# -----------------------------------------------------------------------------
# Optional: Email notifications
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Print GPU and environment info for debugging
# -----------------------------------------------------------------------------
module load cuda/12.4
echo "==============================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==============================="
nvidia-smi
echo "==============================="

# -----------------------------------------------------------------------------
# (1) Load modules or activate environment
# -----------------------------------------------------------------------------
# Example commands (customize for your cluster):
# module load cuda/11.7
# module load anaconda/2023
# source activate nnunet_env

# -----------------------------------------------------------------------------
# (2) Run nnUNet training
# -----------------------------------------------------------------------------
# Syntax: nnUNetv2_train <dataset_name_or_id> <configuration> <fold> [--npz]

fold_name="fold_6"
checkpoint_name="best"

nnUNetv2_predict --c -i "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_0_10/val/images" \
                -o "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset901_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/validation_no_mirror" \
                -d 901 \
                -tr autoPET3_Trainer \
                -p nnUNetResEncUNetLPlansMultiTalent \
                -c 3d_fullres \
                -chk "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset901_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/checkpoint_${checkpoint_name}.pth"\
                --save_probabilities --disable_tta



# If you want to train multiple folds in one script, you could do something like:
# for FOLD in 0 1 2 3 4; do
#   nnUNetv2_train 888 2d $FOLD --npz
# done
