#cti %%
import os
import numpy as np
import json
import nibabel as nib
import pickle
from pathlib import Path
from scipy.ndimage import zoom


def resample_data_or_seg_to_shape(
    data,
    new_shape,
    current_spacing=None,
    new_spacing=None,
    is_seg=True,
    order=1,
    order_z=0,
    force_separate_z=None
):
    if data.ndim != 4:
        raise ValueError(f'Expected data shape (C, Z, Y, X), got {data.shape}')
    zoom_factors = [1.0] + [n / o for n, o in zip(new_shape, data.shape[1:])]
    interp_order = 0 if is_seg else order
    return zoom(data, zoom=zoom_factors, order=interp_order)

# import load_pickle
def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def nifti_seg_to_npy(
    nifti_seg_path : str,
    pkl_path       : str,
    output_dir     : str,
    verbose        : bool = True
):
    pkl_path   = Path(pkl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_name = pkl_path.stem

    # ── Each case gets its OWN properties from its OWN pkl ────
    # e.g. case_001 might have shape (263,256,256) spacing [3.26, 2.73, 2.73]
    #      case_002 might have shape (187,512,512) spacing [5.00, 1.17, 1.17]
    #      case_003 might have bbox  [[10,200],[5,250],[0,256]] (was cropped)
    # All handled automatically below
    props = load_pkl(pkl_path)

    original_spacing = props['spacing']
    bbox             = props['bbox_used_for_cropping']
    shape_after_crop = props['shape_after_cropping_and_before_resampling']

    if verbose:
        print(f"\nCase             : {case_name}")
        print(f"Original spacing : {original_spacing}")   # unique per case
        print(f"BBox             : {bbox}")               # unique per case
        print(f"Shape after crop : {shape_after_crop}")   # unique per case

    # ── Target shape comes from THIS case's image .npy ────────
    # So resampling target is always correct for each individual case
    img_npy_path = pkl_path.with_suffix('.npy')
    if not img_npy_path.exists():
        raise FileNotFoundError(f"Image .npy not found: {img_npy_path}")

    img_data     = np.load(img_npy_path, mmap_mode='r')
    target_shape = img_data.shape[1:]   # THIS case's preprocessed shape

    if verbose:
        print(f"Target shape     : {target_shape}")       # unique per case

    # ── Load NIfTI seg ────────────────────────────────────────
    nib_img = nib.load(nifti_seg_path)
    seg     = nib_img.get_fdata()

    # ── Transpose ─────────────────────────────────────────────
    seg = seg.transpose(2, 1, 0)        # (x,y,z) → (z,y,x)
    seg = seg[np.newaxis]               # → (1, Z, Y, X)

    # ── Crop using THIS case's bbox ───────────────────────────
    # Some cases will have real crops, some won't (like your example)
    z0, z1 = bbox[0]
    y0, y1 = bbox[1]
    x0, x1 = bbox[2]
    seg = seg[:, z0:z1, y0:y1, x0:x1]

    assert list(seg.shape[1:]) == list(shape_after_crop), (
        f"Shape mismatch after crop!\n"
        f"  Got      : {seg.shape[1:]}\n"
        f"  Expected : {shape_after_crop}\n"
    )

    # ── Resample to THIS case's target shape ──────────────────
    # Uses THIS case's original_spacing for correct mm-space resampling
    if list(seg.shape[1:]) != list(target_shape):
        if verbose:
            print(f"Resampling {seg.shape[1:]} → {target_shape}")

        seg = resample_data_or_seg_to_shape(
            data             = seg,
            new_shape        = target_shape,
            current_spacing  = original_spacing,  # THIS case's spacing
            new_spacing      = None,
            is_seg           = True,
            order            = 1,
            order_z          = 0,
            force_separate_z = None
        )
    else:
        if verbose:
            print("No resampling needed — shapes already match")

    # ── Cast dtype ────────────────────────────────────────────
    if np.max(seg) > 127:
        seg = seg.astype(np.int16)
    else:
        seg = seg.astype(np.int8)

    # ── Save ──────────────────────────────────────────────────
    output_path = output_dir / f"{case_name}_seg.npy"
    np.save(output_path, seg)
    print(f"✓ Saved → {output_path}")

    return seg


def batch_convert(
    nifti_labels_dir : str,
    pkl_dir          : str,
    output_dir       : str,
    verbose          : bool = False
):
    nifti_labels_dir = Path(nifti_labels_dir)
    pkl_dir          = Path(pkl_dir)

    seg_files = sorted(nifti_labels_dir.glob("*.nii.gz"))
    print(f"Found {len(seg_files)} segmentation files\n")

    # Print a preview so you can see the variation across cases
    print("Case preview (first 5):")
    for seg_path in seg_files[:5]:
        case_name = seg_path.name.replace(".nii.gz", "")
        pkl_path  = pkl_dir / f"{case_name}.pkl"
        if pkl_path.exists():
            props = load_pkl(pkl_path)
            print(f"  {case_name}")
            print(f"    spacing : {props['spacing']}")
            print(f"    bbox    : {props['bbox_used_for_cropping']}")
            print(f"    shape   : {props['shape_after_cropping_and_before_resampling']}")
    print()

    failed = []

    for seg_path in tqdm(seg_files):
        case_name = seg_path.name.replace(".nii.gz", "")
        pkl_path  = pkl_dir / f"{case_name}.pkl"

        if not pkl_path.exists():
            print(f"[WARN] No pkl for {case_name} — skipping")
            failed.append(case_name)
            continue

        output_seg = Path(output_dir) / f"{case_name}_seg.npy"
        if output_seg.exists():
            print(f"[SKIP] {case_name} already done")
            continue

        try:
            nifti_seg_to_npy(
                nifti_seg_path = str(seg_path),
                pkl_path       = str(pkl_path),
                output_dir     = output_dir,
                verbose        = verbose
            )
        except Exception as e:
            print(f"[ERROR] {case_name}: {e}")
            failed.append(case_name)

    print(f"\n✓ Done: {len(seg_files)-len(failed)}/{len(seg_files)} converted")
    if failed:
        print(f"  Failed: {failed}")

# %%
import os
import numpy as np
import nibabel as nib
import pickle
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────

import os
import argparse
argparse = argparse.ArgumentParser(description='Convert NIfTI segmentations to NumPy arrays with resampling.')
argparse.add_argument('--nifti_labels_dir', type=str, required=True, help='Directory containing NIfTI segmentation files (.nii.gz)')
argparse.add_argument('--pkl_dir', type=str, required=True, help='Directory containing .pkl files with case properties')
argparse.add_argument('--output_dir', type=str, required=True, help='Directory to save the converted .npy segmentation files')
args = argparse.parse_args()





# # ── Full dataset ──────────────────────────────────────────────
batch_convert(
    nifti_labels_dir = args.nifti_labels_dir,
    pkl_dir          = args.pkl_dir,
    output_dir       = args.output_dir,
    verbose          = False
)


