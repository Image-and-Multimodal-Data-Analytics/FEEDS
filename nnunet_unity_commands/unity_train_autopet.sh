#!/bin/bash

#SBATCH --job-name=999_10
#SBATCH --output=autopet3_train_%j.out
#SBATCH --error=autopet3_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=40G

# Clean caches
rm -rf /scratch/f007g3j/torchinductor_f007g3j/

# Safe env
export nnUNet_compile=False
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
export CUDA_LAUNCH_BLOCKING=1     # keep while debugging

# Re-run
# nnUNetv2_train 902 3d_fullres 1 -tr autoPET3_Trainer \
#     -p nnUNetResEncUNetLPlansMultiTalent \
#     -pretrained_weights .../checkpoint_final.pth --npz
#
# 6) Confirm which Python is being used
echo "Using Python at: $(which python)"



#This is for training with pretrained with uncertainty maps
nnUNetv2_train 999 3d_fullres 10 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights  ~/pretrained_final_checkpoint.pth --npz 

# nnUNetv2_train 111 3d_fullres 7 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent --npz --c


echo "Training completed at: $(date)"
