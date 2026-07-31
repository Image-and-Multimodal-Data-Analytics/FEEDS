#!/bin/bash 

# -----------------------------------------------------------------------------
# Basic Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --job-name=fold_2_testing    # Descriptive job name
#SBATCH --account=free
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq   # Partition (queue) name on your cluster
#SBATCH --time=3-00:00:00              # Walltime (HH:MM:SS) 
#SBATCH --mem=32G                    # Memory request: 16 GB

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
#!/bin/bash

BASE="/mnt/z/Results/nnUNet_data"

dataset_name="Dataset902_AutoPet"
# nnUNetv2_predict --c -i "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/train/imagesTr" \
#                -o "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/train_predictions" \
for fold_name in fold_0 fold_1 fold_2; do
    nnUNetv2_predict --c -i "${BASE}/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_2_mg/test/images" \
                    -o "${BASE}/nnUNet_results/Dataset902_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/test_predictions" \
                    -d 902 \
                    -tr autoPET3_Trainer \
                    -p nnUNetResEncUNetLPlansMultiTalent \
                    -c 3d_fullres \
                    -chk "${BASE}/nnUNet_results/Dataset902_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold_name}/checkpoint_best.pth" --save_probabilities 
done
# If you want to train multiple folds in one script, you could do something like:
# for FOLD in 0 1 2 3 4; do
#   nnUNetv2_train 888 2d $FOLD --npz
# done