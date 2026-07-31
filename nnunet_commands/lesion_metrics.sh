#!/bin/bash
#SBATCH -J 34c
#SBATCH --partition=l40s_indrani
#SBATCH --gres=gpu:1
#SBATCH -o filename_%j.out          # File to save standard output
#SBATCH -e filename_%j.err          # File to save standard error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7:00:00
#SBATCH --mem=40G
#SBATCH --account=bhattacharya-lab
# Print GPU status and hostname (these will be logged in the output file)
nvidia-smi
hostname

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# loop through the folds
basedir="//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/"
outdir="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/lesion_metrics/autoPet_3_Trainer__nnUNetResEncUNetLPlansMultiTalent_3d_fullres"

seconddir="autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_"
folds=('first')

for fold in "${folds[@]}"
do
    preddir="${basedir}/${seconddir}${fold}/test_predictions/"
    python autopet3_lesion_metrics.py -f $fold -p "$preddir" -o "$outdir"
done
