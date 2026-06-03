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
    parser.add_argument('-i', '--image-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/imagesTr/", help='Directory containing images (default: ../imagesTr/)')
    parser.add_argument('-m', '--mets-only', action='store_true', 
                       help='Process only mets data, skip lesion-only data')
    parser.add_argument('-ld', '--label-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/labelsTr/", help='Directory containing labels (default: ../labelsTr/)') 
    parser.add_argument('-b', '--bone-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/bone-metastasis/totalsegmentator/bone_segmentations_fold1/", help='Directory containing bone segmentations (default: ../totalsegmentator/bone_segmentations_fold1/)')

    parser.add_argument('-p', '--pred-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset666_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_epi_no_norm_unc_input_ens_comb/test_predictions/", 
                        help='Directory containing prediction files (default: //dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/test_predictions/)')

    parser.add_argument('-u', '--uncertainty-dir', type=str, default="//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/uncertainty_maps/", 
                        help='Directory containing uncertainty maps (default: //dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset888_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_1_ens_comb/uncertainty_maps/)')
    parser.add_argument('-o', '--save-dir', type=str, default="./lesion_analysis_fold", 
                        help='Directory to save lesion analysis results (default: ./lesion_analysis_fold)')
    parser.add_argument("--organ_map", type=str, default="./ts_table.csv", help="CSV file with columns: Label, Description, is_bone")
    args = parser.parse_args()
    return args

def main():
    """ 
    Driver function to process images based on command line arguments.
    """    
     
    args = handle_arguments()
    ONLY_LESION_STATUS = args.lesion_only
    fold = args.fold
    k = args.key


    global LESION_SAVE_FOLDER_DIR
    LESION_SAVE_FOLDER_DIR = f"./lesion_analysis_fold{fold}"
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
    
    if ONLY_LESION_STATUS:
        print(f"Processing only lesion data...", flush=True)
    else:
        print(f"Processing full CTImage object...", flush=True)
    
    print(f"Using fold: {fold}", flush=True)
    print(f"Skip existing files: {args.skip_existing}", flush=True)
    
    print("Directories set:", flush=True)
    for key, path in dirs.items():
        print(f"  {key}: {path}", flush=True)

    save_file = f"./summary_{k.replace('*', '')}.pkl"
    print(f"Summary will be saved to: {save_file}", flush=True)

    # Load existing summary
    existing_summary = load_summary_data(save_file)
    already_processed = set()
    if existing_summary:
        already_processed = {data['id'] for data in existing_summary}
        print(f"Found {len(already_processed)} already processed images", flush=True)
    
    # Process files
    counter = 0
    skipped = 0
    
    # Get all lesion files first (these are our base IDs)
    lesion_files = sorted(glob.glob(dirs['labels'] + f"{k}*.nii.gz"))
    print(f"Found {len(lesion_files)} lesion files to process", flush=True)
    
    for lesion_file in lesion_files:
        # Extract base lesion ID
        lesion_id = os.path.basename(lesion_file).split('.')[0]
        print(f"Looking for matching files for lesion ID: {lesion_id}", flush=True)
        
        # Find matching files using our improved functions
        image_file = find_matching_file(lesion_id, dirs['images'], "*.nii.gz")
        bone_file = find_matching_file(lesion_id, dirs['bone'], "*ts.nii.gz")
        pred_file = find_matching_file(lesion_id, dirs['pred'], "*.nii.gz")
        
        # Check if all required files are found
        if image_file and bone_file and pred_file:
            print(f"Found all matching files for {lesion_id}:", flush=True)
            print(f"  Image: {os.path.basename(image_file)}", flush=True)
            print(f"  Lesion: {os.path.basename(lesion_file)}", flush=True)
            print(f"  Bone: {os.path.basename(bone_file)}", flush=True)
            print(f"  Pred: {os.path.basename(pred_file)}", flush=True)
            
            image_files = {
                'image': image_file,
                'lesion': lesion_file,
                'bone': bone_file,
                'pred': pred_file
            }

            existing_summary = process_single_image(image_files, lesion_id, dirs, existing_summary, ONLY_LESION_STATUS)
            counter += 1
            # IF only lesion data, no need to save 
            if ONLY_LESION_STATUS:
                print(f"Processed lesion data for {lesion_id}, skipping CTImage object creation.", flush=True)
                continue

            # Save periodically
            if counter % 5 == 0:
                with open(save_file, 'wb') as f:
                    pickle.dump(existing_summary, f)
                print(f"Saved summary after processing {counter} files", flush=True)
        else:
            skipped += 1
            missing_files = []
            if not image_file: missing_files.append("image")
            if not bone_file: missing_files.append("bone")
            if not pred_file: missing_files.append("prediction")
            print(f"Skipping {lesion_id} - missing files: {', '.join(missing_files)}", flush=True)
    
    # Final save
    with open(save_file, 'wb') as f:
        pickle.dump(existing_summary, f)
    print(f"Final save completed. Processed {counter} new images, skipped {skipped}", flush=True)

if __name__ == "__main__":
    main()
