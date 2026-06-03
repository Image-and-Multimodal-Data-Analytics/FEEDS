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
from probability_analysis import handle_arguments
from process import process_bone_mets

import argparse
from tqdm import tqdm

args = handle_arguments()
fold = args.fold


dirs = {
    'images': args.image_dir,
    'labels': args.label_dir, 
    'bone': args.bone_dir,
    'pred': args.pred_dir,
}

print(dirs['images'])
print(dirs['labels'])

global LESION_SAVE_FOLDER_DIR

k = args.key
print(k)

LESION_SAVE_FOLDER_DIR = os.path.join(args.save_dir, str(fold))


os.makedirs(LESION_SAVE_FOLDER_DIR, exist_ok=True)
print(f"Lesion analysis results will be saved in: {LESION_SAVE_FOLDER_DIR}", flush=True)
############### Printing Statements ###############

print(f"Processing only mets data...", flush=True)
print(f"Using fold: {fold}", flush=True)
print("Directories set:", flush=True)

for key, path in dirs.items():
    print(f"  {key}: {path}", flush=True)

############### Load existing summary #############
counter = 0
skipped = 0

# Get all lesion files first (these are our base IDs)
image_files = sorted(glob.glob(os.path.join(dirs['labels'], f"{k}*.nii.gz")))
print(f"Found {len(image_files)} image files to process", flush=True)

###################  Analysis  ####################
for image_file in tqdm(image_files, desc="Processing images"):
    #print(f"Processing image file: {image_file}", flush=True)
    # Extract base lesion ID
    image_id = os.path.basename(image_file).split('.')[0]

    # check if the file is already processed
    if os.path.exists(f"{LESION_SAVE_FOLDER_DIR}/{image_id}_mets.csv"):
        print(f"File {image_id}_mets.csv already exists. Skipping...", flush=True)
        skipped += 1
        continue

    bone_mets_obj = process_bone_mets(image_file, dirs)

    bone_mets_obj.merged_table.to_csv(f"{LESION_SAVE_FOLDER_DIR}/{image_id}_mets.csv", index=False)
    bone_mets_obj.metrics.to_csv(f"{LESION_SAVE_FOLDER_DIR}/{image_id}_mets_metrics.csv", index=False)
