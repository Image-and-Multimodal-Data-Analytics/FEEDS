import json
import os


print(os.getcwd())

SPLITS_DIR = '/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/Dataset999_AutoPet/splits_final.json'
TRAIN_FILES_DIR = '/dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Autopet_III_nnunet_raw/Dataset888_AutoPet/labelsTr'
TRAIN_LINK_DIR = '/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/hard_labels/'
# Load the json



with open(SPLITS_DIR, 'r') as f:
    splits = json.load(f)

# Get labelled and all filenames
labelled = set(splits[0]['train'])
all_files = set(splits[6]['train'])

# Get unlabelled only
unlabelled = all_files - labelled

# # Base directory where the actual files are located
base_dir = TRAIN_FILES_DIR
# os.listdir(base_dir)
# # Create the output folder
output_dir = TRAIN_LINK_DIR
os.makedirs(output_dir, exist_ok=True)

# # Create symlinks

def fname_is_image(fname): 
    fname_pet = fname + '_0000.nii.gz'  # <-- CHANGE EXTENSION IF NEEDED
    fname_ct = fname + '_0001.nii.gz'  #
    # Adjust extension if needed

    src = os.path.join(base_dir, fname_pet)  # Assuming images are in base_dir
    dst = os.path.join(output_dir, fname_pet)  # Link will be created in output_dir    

    if os.path.exists(src):
        os.symlink(src, dst)
    else:
        print(f'File not found: {os.path.basename(src)}')

    src_2 = os.path.join(base_dir, fname_ct)  # Assuming images are in base_dir
    dst_2 = os.path.join(output_dir, fname_ct)  # Link will be created in output_dir

    if os.path.exists(src_2):
        os.symlink(src_2, dst_2)
    else:
        return(f'File not found: {os.path.basename(src_2)}')

def fname_is_label(fname): 
    fname_label = fname + '.nii.gz'  # <-- CHANGE EXTENSION IF NEEDED

    src = os.path.join(base_dir, fname_label)  # Assuming labels are in base_dir
    dst = os.path.join(output_dir, fname_label)  # Link will be created in output_dir    

    if os.path.exists(src):
        os.symlink(src, dst)
    else:
        print(f'File not found: {os.path.basename(src)}')


for fname in labelled: 
    fname_is_label(fname)
    # fname_is_image(fname)
print(f'Done! Created {len(labelled)} symlinks in {output_dir}')