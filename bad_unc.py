# for file in bad_uncertainty files 
# get the OG dir for them and make thm 

import os
import pandas as pd

bad_uncertainty_files = pd.read_csv('bad_uncertainty_files.csv')

ORIGNAL_TRAIN_LINK_DIR = '//dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Autopet_III_nnunet_raw/Dataset888_AutoPet/imagesTr'
BAD_FILES_SUBSET_DIR = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/bad_files_subset'

os.makedirs(BAD_FILES_SUBSET_DIR, exist_ok=True)



for file in bad_uncertainty_files['filename']:
    pet_dir = os.path.join(ORIGNAL_TRAIN_LINK_DIR, file.replace('.nii.gz', '_0001.nii.gz'))
    ct_dir = os.path.join(ORIGNAL_TRAIN_LINK_DIR, file.replace('.nii.gz', '_0000.nii.gz'))
    print(pet_dir)
    print(os.path.exists(pet_dir))
    print(ct_dir)
    print(os.path.exists(ct_dir))
    
    # if both exist, symlink them to the BAD_FILES_SUBSET_DIR
    if os.path.exists(pet_dir) and os.path.exists(ct_dir):
        try: 
            os.symlink(pet_dir, os.path.join(BAD_FILES_SUBSET_DIR, os.path.basename(pet_dir)))
            os.symlink(ct_dir, os.path.join(BAD_FILES_SUBSET_DIR, os.path.basename(ct_dir)))
        except FileExistsError:
            print(f"Symlink already exists for {file}, skipping.")
    print("-" * 50)
    print(f"Processed file: {file}")
