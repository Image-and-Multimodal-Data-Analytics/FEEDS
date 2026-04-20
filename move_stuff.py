import json
import os
import shutil
from pathlib import Path

os.chdir('//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/')
# print the listdir
print(f"📂 Contents of current directory: {os.listdir()}")
print(f"📂 Current working directory: {os.getcwd()}")
# Read splits for 10 percent
with open('./nnUNet_processed/Dataset999_AutoPet/splits_final_10_percent.json', 'r') as f:
    data = json.load(f)

hard_masks = [x + ".nii.gz" for x in data[0]['train']]

# directory of original masks
original_masks_dir  = "//dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Autopet_III_nnunet_raw/Dataset888_AutoPet/labelsTr"
predicted_masks_dir = "//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_10/train_predictions/"

original_mask_files  = set(os.listdir(original_masks_dir))
predicted_mask_files = set(x for x in os.listdir(predicted_masks_dir) if x.endswith(".nii.gz"))

print(f"📂 Hard masks in json       : {len(hard_masks)}")
print(f"📂 Original masks on disk   : {len(original_mask_files)}")
print(f"📂 Predicted masks on disk  : {len(predicted_mask_files)}")

# find the intersection
intersection = set(hard_masks) & original_mask_files & predicted_mask_files

# Show what's missing
missing_from_original  = set(hard_masks) - original_mask_files
missing_from_predicted = set(hard_masks) - predicted_mask_files

if missing_from_original:
    print(f"\n⚠️  {len(missing_from_original)} hard masks missing from original_masks_dir:")
    for f in sorted(missing_from_original):
        print(f"   ❌ {f}")

if missing_from_predicted:
    print(f"\n⚠️  {len(missing_from_predicted)} hard masks missing from predicted_masks_dir:")
    for f in sorted(missing_from_predicted):
        print(f"   ❌ {f}")

hard_mask_dirs       = [os.path.join(original_masks_dir,  x) for x in intersection]
predicted_train_dirs = [os.path.join(predicted_masks_dir, x) for x in intersection]

print(f"\n✅ Intersection (will process): {len(intersection)} files\n")

# ── Backup predicted files that will be overwritten ───────────────────────────
PREDICTED_TRAIN_DIR = Path(predicted_masks_dir) / "predicted_trained"
PREDICTED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

moved  = 0
failed = 0

for prediction_on_train in predicted_train_dirs:
    src = Path(prediction_on_train)
    dst = PREDICTED_TRAIN_DIR / src.name

    if src.is_file():
        shutil.move(str(src), str(dst))
        print(f"📦 Moved   : {src.name} → predicted_trained/")
        moved += 1
    else:
        print(f"❌ NOT FOUND: {src}")
        failed += 1

print(f"\n📦 Moved {moved} files, {failed} failed\n")

# ── Copy hard masks into predicted_masks_dir ──────────────────────────────────
copied = 0
failed = 0

for hard_mask in hard_mask_dirs:
    src = Path(hard_mask)
    dst = Path(predicted_masks_dir) / src.name

    if src.is_file():
        shutil.copy2(str(src), str(dst))
        print(f"✅ Copied   : {src.name} → predicted_masks_dir/")
        copied += 1
    else:
        print(f"❌ NOT FOUND: {src}")
        failed += 1

print(f"\n✅ Copied {copied} files, {failed} failed")