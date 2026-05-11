import json
import os
import numpy as np

print(os.getcwd())

# ── Paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR        = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/{dataset}/splits_final.json'
DATASET_ID        = 901   # <-- CHANGE AS NEEDED

SPLITS_DIR = SPLITS_DIR.format(dataset=f'Dataset{DATASET_ID:03d}_AutoPet')
# Source directories
HARD_LABELS_DIR    = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/Dataset999_AutoPet/nnUNetPlans_3d_fullres/'
PSEUDO_LABELS_DIR = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_0_10/train_tta_predicted/train_predictions_old/converted_segs'
VAL_LABELS_DIR    = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/Dataset999_AutoPet/nnUNetPlans_3d_fullres/'
TEST_LABELS_DIR  = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_0_10/test/'

# Destination
NNUNET_RAW_DIR    = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/'
NEW_DATASET_DIR   = os.path.join(NNUNET_RAW_DIR, f'Dataset{DATASET_ID:03d}_AutoPet')
OUTPUT_LABELS_DIR = os.path.join(NEW_DATASET_DIR, 'nnUNetPlans_3d_fullres')

# ── Load splits ────────────────────────────────────────────────────────────────
with open(SPLITS_DIR, 'r') as f:
    splits = json.load(f)


splits_from_888 = SPLITS_DIR.replace(f'Dataset{DATASET_ID:03d}_AutoPet', 'Dataset888_AutoPet')
with open(splits_from_888, 'r') as f:
    splits_888 = json.load(f)

    
labelled   = set(splits[0]['train'])
all_files  = set(splits_888[1]['train'])
unlabelled = all_files - labelled
val_files  = set(splits[0]['val'])

all_hard_labels =[f for f in os.listdir(HARD_LABELS_DIR) if f.endswith('_seg.npy')]

print(f"Total labelled cases: {len(labelled)}")
print(f"Total unlabelled cases: {len(unlabelled)}")
print(f"Total validation cases: {len(val_files)}")

print(len(all_hard_labels))


print("first few entries in labelled:")
print(list(labelled)[:2])
# remove the unlabelled files from the hard labels list
all_hard_labels = [f for f in all_hard_labels if f.replace('_seg.npy', '') not in unlabelled]
print(len(all_hard_labels))

# assert that the directories exist and print the first 2 entries in it
assert os.path.exists(HARD_LABELS_DIR), f"Directory does not exist: {HARD_LABELS_DIR}"
assert os.path.exists(PSEUDO_LABELS_DIR), f"Directory does not exist: {PSEUDO_LABELS_DIR}"
assert os.path.exists(TEST_LABELS_DIR), f"Directory does not exist: {TEST_LABELS_DIR}"

# ── Create output directory ────────────────────────────────────────────────────
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
print(f'Created dataset directory: {NEW_DATASET_DIR}')

# ── Helper ─────────────────────────────────────────────────────────────────────
def make_symlink(src: str, dst: str, force: bool = False) -> None:
    """
    Create a symlink at `dst` pointing to `src`.
    Skips gracefully if the source does not exist or the link already exists.
    """
    if not os.path.exists(src):
        print(f'  [MISSING]  {os.path.basename(src)}')
        return
    if os.path.exists(dst) or os.path.islink(dst):
        if force:
            print(f'  [FORCE]   {os.path.basename(dst)}')
            os.remove(dst)
        else:
            print(f'  [EXISTS]  {os.path.basename(dst)}')
            return
        
    os.symlink(src, dst)

def link_label(fname: str, source_dir: str, force: bool = False) -> None:
    """Symlink a single label file from `source_dir` into OUTPUT_LABELS_DIR."""

    filename = fname 
    destination_file_name = fname 
    
    src = os.path.join(source_dir, filename)
    dst = os.path.join(OUTPUT_LABELS_DIR, destination_file_name)
    make_symlink(src, dst, force=force)

total = 0 

print('\n── Linking pseudo labels ──')
print(len(unlabelled))



for fname in labelled:
    fname = fname + '_seg.npy'
    link_label(fname, HARD_LABELS_DIR, force=True)

print(f'  Processed {len(labelled)} hard label(s).')
total += len(labelled)

for fname in val_files: 
    fname = fname + '_seg.npy'
    link_label(fname, HARD_LABELS_DIR, force=True)
print(f'  Processed {len(val_files)} validation label(s).')
total += len(val_files)

for fname in unlabelled:
    fname = fname + '_seg.npy'  
    # check if the file exists already
    link_label(fname, PSEUDO_LABELS_DIR, force=True)

#  ── 2. Pseudo labels (unlabelled cases) ───────────────────────────────────────
# # anything that is in hardlabels dir and hasn't been linked yet, link it here
# print('\n── Linking remaining hard labels ──')


# # ── Summary ───────────────────────────────────────────────────────────────────

print(f'\nDone! Created up to {total} symlink(s) in:\n  {OUTPUT_LABELS_DIR}')
print(f"PROCESSED {len(labelled)} labelled cases")
print(f"PROCESSED {len(unlabelled)} unlabelled cases")
print(f"PROCESSED {len(val_files)} validation cases")