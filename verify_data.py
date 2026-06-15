from asyncio import sleep
import json 
import numpy as np
import os
import nibabel as nib

# This is the directory I'm going to use for training
Dataset = 223

DIR_TO_CHECK_DIR = f'{os.getenv("nnUNet_preprocessed")}/Dataset{Dataset}_AutoPet/nnUNetPlans_3d_fullres/'
HARD_LABELS_DIR    = '//{nnUNet_preprocessed}/Dataset999_AutoPet/nnUNetPlans_3d_fullres/'.format(nnUNet_preprocessed=os.getenv('nnUNet_preprocessed'))

PSEUDO_LABELS_DIR = '//{nnUNet_results}/Dataset111_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_8/train_predictions/converted_segs'.format(nnUNet_results=os.getenv('nnUNet_results'))
TEST_LABELS_DIR  = f'{os.getenv("nnUNet_results")}/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_0_10/test/gt'
# print cuda version too 
SPLITS_DIR        = '//{nnUNet_preprocessed}/{dataset}/splits_final.json'.format(nnUNet_preprocessed=os.getenv('nnUNet_preprocessed'), dataset='Dataset999_AutoPet')

HARD_LABLLED_SPLITS = '//{nnUNet_preprocessed}/Dataset111_AutoPet/splits_final.json'.format(nnUNet_preprocessed=os.getenv('nnUNet_preprocessed'))

with open(SPLITS_DIR, 'r') as f:
    splits_for_all = json.load(f)

with open(HARD_LABLLED_SPLITS, 'r') as f:
    splits = json.load(f)
    



for d in [DIR_TO_CHECK_DIR, HARD_LABELS_DIR, PSEUDO_LABELS_DIR, HARD_LABELS_DIR, TEST_LABELS_DIR, HARD_LABELS_DIR]:
    print(f"Checking directory: {d}")
    if os.path.exists(d):
        print(f" EXISTS: {d}")
        files = os.listdir(d)
        print(f"\t Number of files: {len(files)}")
    else:
        print(f"NOT EXIST: {d}")


labelled   = set(splits[8]['train'])
all_files  = set(splits_for_all[-1]['train'])
unlabelled = all_files - labelled
val_files  = set(splits[0]['val'])

print(f"Total labelled cases: {len(labelled)}")
print(f"Total unlabelled cases: {len(unlabelled)}")
print(f"Total validation cases: {len(val_files)}")

def check_file_exists(fname, directory):
    path = os.path.join(directory, fname + '_seg.npy')
    if os.path.exists(path):
        return True
    else:
        print(f"  MISSING: {path}")
        return False

def check_if_same_file(fname, dir1, dir2):
    path1 = os.path.join(dir1, fname + '_seg.npy')
    path2 = os.path.join(dir2, fname + '_seg.npy')
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"  Cannot compare. One of the files is missing:")
        return False
    
    img1 = np.load(path1)
    img2 = np.load(path2)
    
    if np.array_equal(img1, img2):
        print(f"  MATCH: {fname} in both directories are the same.")
        return True
    else:
        print(f"  MISMATCH: {fname} differs between the two directories.")
        # print the shape 
        print(f"    Shape in {dir1}: {img1.shape}")
        print(f"    Shape in {dir2}: {img2.shape}")
        return False
    
print("")
print("=== CHECKING VALIDATION FILES ===")
print("")
# load a random validation file from the val_labels directory and check if it matches the name of the file in the splits
val_file = list(val_files)[0]
val_file_path = os.path.join(HARD_LABELS_DIR, val_file + '_seg.npy')
check_val_file_path = os.path.join(DIR_TO_CHECK_DIR, val_file + '_seg.npy')


val_status = False
test_status = False
hard_label_status = False
pseudo_label_status = False


print(f"Checking validation file: {val_file_path}")
if check_file_exists(val_file, HARD_LABELS_DIR) and check_file_exists(val_file, DIR_TO_CHECK_DIR):
    print("Both validation files exist. Now checking if they are the same...")
    val_status = check_if_same_file(val_file, HARD_LABELS_DIR, DIR_TO_CHECK_DIR)



print(f"Checking if a hard label file exists in the directory to check: {DIR_TO_CHECK_DIR}")
# check for all hard files 
hard_label_status_list = []
count = 0
for hard_label_file in labelled:
    hard_label_path = os.path.join(HARD_LABELS_DIR, hard_label_file + '_seg.npy')
    check_hard_label_path = os.path.join(DIR_TO_CHECK_DIR, hard_label_file + '_seg.npy')
    if check_file_exists(hard_label_file, HARD_LABELS_DIR) and check_file_exists(hard_label_file, DIR_TO_CHECK_DIR):
        hard_label_status = check_if_same_file(hard_label_file, HARD_LABELS_DIR, DIR_TO_CHECK_DIR)
        hard_label_status_list.append(hard_label_status)
    count += 1
    if count >= 170:  # Limit to first 10 files
        break

# if any are different, the overall status is different
hard_label_status = all(hard_label_status_list) if hard_label_status_list else False

print(f"Checking if a pseudo label file exists in the directory to check: {DIR_TO_CHECK_DIR}")

pseudo_label_status_list = []
count = 0
for pseudo_label_file in unlabelled:
    pseudo_label_path = os.path.join(HARD_LABELS_DIR, pseudo_label_file + '_seg.npy')
    check_pseudo_label_path = os.path.join(DIR_TO_CHECK_DIR, pseudo_label_file + '_seg.npy')

    if check_file_exists(pseudo_label_file, HARD_LABELS_DIR) and check_file_exists(pseudo_label_file, DIR_TO_CHECK_DIR):
        pseudo_label_status = check_if_same_file(pseudo_label_file, HARD_LABELS_DIR, DIR_TO_CHECK_DIR)
        pseudo_label_status_list.append(pseudo_label_status)
    count += 1
    if count >= 10:  # Limit to first 10 files
        break
pseudo_label_status = all(pseudo_label_status_list) if pseudo_label_status_list else True

# Final summary
print("\n=== SUMMARY ===")
print(f"Validation file check: {'SAME' if val_status else 'DIFFERENT'}")
print(f"Hard label file check: {'SAME' if hard_label_status else 'DIFFERENT'}")
print(f"Pseudo label file check: {'SAME' if pseudo_label_status else 'DIFFERENT'}")
