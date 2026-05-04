#!/bin/bash

#SBATCH --job-name=qq_9
#SBATCH --output=autopet3_train_%j.out
#SBATCH --error=autopet3_train_%j.err
#SBATCH --account=bhattacharya-lab
# #SBATCH --nodelist=adanova01
#SBATCH --partition=l40s_indrani
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=40G

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$(whoami)_$$
export TRITON_CACHE_DIR=/tmp/triton_$(whoami)_$$
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCH_COMPILE_DISABLE=1

#
# 6) Confirm which Python is being used
echo "Using Python at: $(which python)"


#This is for training with pretrained 
#nnUNetv2_train 888 3d_fullres 1 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /dartfs/rc/lab/B/BhattacharyaI/Results/Bashirul/Research/autopet-3-submission/pretrained/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth --npz 


#This is for training with pretrained with uncertainty maps
# nnUNetv2_train 901 3d_fullres 10 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights  /dartfs/rc/lab/B/BhattacharyaI/Results/Bashirul/Research/autopet-3-submission/pretrained/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth --npz 

nnUNetv2_train 901 3d_fullres 9 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent --npz --c

#This for training with last epoch 
# nnUNetv2_train 999 3d_fullres 1 -tr autoPET3_Trainer -p nnUNetResEncUNetLPlansMultiTalent --c --npz 
echo "Training completed at: $(date)"