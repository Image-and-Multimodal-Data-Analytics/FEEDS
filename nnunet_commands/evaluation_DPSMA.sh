#!/bin/bash

# -----------------------------------------------------------------------------
# Basic Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --job-name=evaluate_nnunet_validation    # Descriptive job name
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --account=free
# #SBATCH --account=free
#SBATCH --ntasks-per-node=1          # Number of tasks per node
# #SBATCH --gres=gpu:1
# #SBATCH --partition=        # Partition (queue) name on your cluster
#SBATCH --partition=gpuq       # Partition (queue) name on your cluster
#SBATCH --time=00:20:00              # Walltime (HH:MM:SS) 
#SBATCH --mem=16G                    # Memory request: 16 GB
# -----------------------------------------------------------------------------
# Print GPU and environment info for debugging
# -----------------------------------------------------------------------------
# module load cuda/12.4
echo "==============================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==============================="
nvidia-smi
echo "==============================="

# -----------------------------------------------------------------------------
# (1) Load modules or activate environmentsq
# -----------------------------------------------------------------------------
# Example commands (customize for your cluster):
# module load cuda/11.7
# module load anaconda/2023
# source activate nnunet_env

# for fold in fold_10_ens_0 fold_10_ens_1 fold_10_ens_2 fold_10_ens_3 fold_10_ens_4; do

input_dataset="Dataset903_AutoPet"
validation="DPSMA_predictions"

for fold in fold_0 fold_1 fold_2; do
  nnUNetv2_evaluate_folder \
    -djfile /dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/dataset.json \
    -pfile /dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/plans.json \
    /dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Deep_PSMA_CT_reshaped/Dataset701_deep_psma/labelsTr \
    /dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/$input_dataset/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/${fold}/${validation}/ 
done