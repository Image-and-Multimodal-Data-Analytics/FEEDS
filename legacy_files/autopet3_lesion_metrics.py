
import os
import glob
import json
import pickle

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from data_structures import * 
from files import find_matching_file, find_matching_files
from summary import * 
from uncertainty import * 
from analysis import * 
import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys

import argparse

def handle_arguments():
    """
    Handle command line arguments for processing images.
    """
    parser = argparse.ArgumentParser(description='Process probability analysis with various options')
    parser.add_argument('-k', '--key', type=str, default='*', 
                       help='Key pattern for file matching (default: *)')
    parser.add_argument('-l', '--lesion-only', action='store_true', 
                       help='Process only lesion data, skip CTImage object creation')
    parser.add_argument('-f', '--fold', type=str, default='comb', 
                       help='Fold name/number (default: 1)')
    parser.add_argument('-s', '--skip-existing', action='store_true',
                       help='Skip processing files that have already been processed', default=False)
    parser.add_argument('-i', '--image-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Autopet_III_nnunet_raw/Dataset888_AutoPet/imagesTr/", help='Directory containing images (default: ../imagesTr/)')
    parser.add_argument('-m', '--mets-only', action='store_true', 
                       help='Process only mets data, skip lesion-only data')
    parser.add_argument('-ld', '--label-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/", help='Directory containing labels (default: ../labelsTr/)') 
    parser.add_argument('-b', '--bone-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/bone-metastasis/totalsegmentator/bone_segmentations_fold1/", help='Directory containing bone segmentations (default: ../totalsegmentator/bone_segmentations_fold1/)')

    parser.add_argument('-p', '--pred-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/test_predictions/", 
                        help='Directory containing prediction files (default: //dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/test_predictions/)')

    parser.add_argument('-u', '--uncertainty-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/uncertainty_maps/", 
                        help='Directory containing uncertainty maps (default: //dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/uncertainty_maps/)')
    parser.add_argument('-o', '--save-dir', type=str, default="./lesion_analysis_fold", 
                        help='Directory to save lesion analysis results (default: ./lesion_analysis_fold)')
    args = parser.parse_args()
    return args


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg

def false_pos_pix_fast(gt_array, pred_array):
    pred_conn_comp = con_comp(pred_array)
    false_pos = 0
    labels = np.arange(1, pred_conn_comp.max()+1)
    for idx in labels:
        comp_mask = pred_conn_comp == idx
        if (comp_mask & gt_array.astype(bool)).sum() == 0:
            false_pos += comp_mask.sum()
    return false_pos


def false_neg_pix_fast(gt_array, pred_array):
    gt_conn_comp = con_comp(gt_array)
    false_neg = 0
    pred_bool = pred_array.astype(bool)
    for idx in range(1, gt_conn_comp.max()+1):
        comp_mask = gt_conn_comp == idx
        if not (comp_mask & pred_bool).any():
            false_neg += comp_mask.sum()
    return false_neg

def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score



def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol

def compute_metrics_fast(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix_fast(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix_fast(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol

import tqdm

args = handle_arguments()
# print arguments
print("Command line arguments:")
for arg in vars(args):
    print(f"  {arg}: {getattr(args, arg)}", flush=True)

ONLY_LESION_STATUS = True
fold = args.fold

global LESION_SAVE_FOLDER_DIR
LESION_SAVE_FOLDER_DIR = args.save_dir if args.save_dir else f"./lesion_metrics_fold{fold}" 
# Define directories
dirs = {
    'images': args.image_dir, 
    'labels': args.label_dir,
    'bone': args.bone_dir,
    'pred': args.pred_dir,
    'uncertainty': args.uncertainty_dir
}
os.makedirs(LESION_SAVE_FOLDER_DIR, exist_ok=True)
print(f"Lesion analysis results will be saved in: {LESION_SAVE_FOLDER_DIR}")

print(os.listdir(dirs['labels']))
print(os.listdir(dirs['bone']))
print(os.listdir(dirs['pred']))
print(os.listdir(dirs['uncertainty']))

k = ''
if ONLY_LESION_STATUS:
    print(f"Processing only lesion data...", flush=True)
else:
    print(f"Processing full CTImage object...", flush=True)

print(f"Using fold: {fold}", flush=True)

print("Directories set:", flush=True)
for key, path in dirs.items():
    print(f"  {key}: {path}", flush=True)

save_file = f"{args.save_dir}/summary_{k.replace('*', '')}.pkl"
print(f"Summary will be saved to: {save_file}", flush=True)



counter = 0
skipped = 0

# Get all lesion files first (these are our base IDs)
lesion_files = sorted(glob.glob(os.path.join(dirs['labels'], f"{k}*.nii.gz")))
print(f"Found {len(lesion_files)} lesion files to process", flush=True)

list_of_img_files = []

for lesion_file in lesion_files:
    # Extract base lesion ID
    image_id = os.path.basename(lesion_file).split('.')[0]
    print(f"Looking for matching files for lesion ID: {image_id}", flush=True)
    
    # Find matching files using our improved functions
    image_file = find_matching_file(image_id, dirs['images'], "*.nii.gz")
    bone_file = find_matching_file(image_id, dirs['bone'], "*ts.nii.gz")
    pred_file = find_matching_file(image_id, dirs['pred'], "*.nii.gz")
    bone_file = None
 
    # Check if all required files are found
    # print the one that is missing 
    assert image_file is not None, f"Image file not found for lesion ID: {image_id}"
    assert pred_file is not None, f"Prediction file not found for lesion ID: {image_id}"
    assert bone_file is not None, f"Bone file not found for lesion ID: {image_id}"
    
    image_files = {
        'lesion': lesion_file,
        'pred': pred_file,
        'bone': bone_file
    }   
        
    list_of_img_files.append(image_files)




big_df = []

# Wrap the loop with tqdm for a proper progress bar
for image_files in tqdm.tqdm(list_of_img_files, desc="Processing images"):
    nii_gt_path, nii_pred_path = image_files['lesion'], image_files['pred']

    nii_gt_path = plb.Path(nii_gt_path)
    nii_pred_path = plb.Path(nii_pred_path)
    dice_sc, false_pos_vol, false_neg_vol = compute_metrics_fast(nii_gt_path, nii_pred_path)

    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    csv_rows = [nii_gt_path.name, dice_sc, false_pos_vol, false_neg_vol]
    
    # make dataframe
    df = pd.DataFrame(columns=csv_header)
    df.loc[0] = csv_rows
    big_df.append(df)   

big_df = pd.concat(big_df, ignore_index=True)


big_df.to_csv(f"{args.save_dir}/lesion_metrics_{fold}.csv", index=False)
