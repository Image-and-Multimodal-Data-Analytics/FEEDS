# process.py
"""
Core processing functions – one image at a time.
Imports are kept lean; heavy arrays are not copied unnecessarily.
"""

import os
import numpy as np
import SimpleITK as sitk

from data_structures import BoneMetastasis, LesionEval, find_matching_file
from analysis import (analyze_predicted_lesions,
                      analyze_ground_truth_lesions,
                      analyze_image)


# ── uncertainty helpers (formerly uncertainty.py) ────────────────────────────

def _load_uncertainty_maps(lesion_id: str, uncertainty_dir: str):
    maps = {}
    for unc_type, pat in [("aleatoric", "*aleatoric*.nii.gz"),
                          ("epistemic", "*epistemic*.nii.gz"),
                          ("total",     "*total*.nii.gz")]:
        path = find_matching_file(lesion_id, uncertainty_dir, pat)
        if path:
            try:
                import nibabel as nib
                maps[unc_type] = nib.load(path).get_fdata()
            except Exception as e:
                print(f"Error loading {unc_type}: {e}")
                maps[unc_type] = None
        else:
            maps[unc_type] = None
    return maps


def _align_uncertainty(unc_data: dict, ref_shape: tuple):
    aligned = {}
    for k, arr in unc_data.items():
        if arr is None:
            aligned[k] = None
        elif arr.shape == ref_shape:
            aligned[k] = arr
        else:
            t = np.transpose(arr, (2, 1, 0))
            aligned[k] = t if t.shape == ref_shape else None
            if aligned[k] is None:
                print(f"Warning: could not align {k} uncertainty")
    return aligned


# ── single-image processing ───────────────────────────────────────────────────

def process_image(lesion_file: str, dirs: dict, only_lesion: bool = False):
    """
    Full lesion-level analysis for one image.

    Returns (pred_stats, ground_stats, image_stats) DataFrames/dicts,
    or None if required files are missing.
    """
    image_id = os.path.basename(lesion_file).split(".")[0]

    image_file = find_matching_file(image_id, dirs["images"], "*.nii.gz")
    bone_file  = find_matching_file(image_id, dirs["bone"],   "*ts.nii.gz")
    pred_file  = find_matching_file(image_id, dirs["pred"],   "*.nii.gz")

    if not (image_file and bone_file and pred_file):
        missing = [n for n, f in [("image", image_file),
                                   ("bone",  bone_file),
                                   ("pred",  pred_file)] if not f]
        print(f"Skipping {image_id} – missing: {', '.join(missing)}")
        return None

    ref_image = sitk.ReadImage(image_file)
    ref_shape = sitk.GetArrayFromImage(ref_image).shape

    # ── probability ────────────────────────────────────────────────────
    probability = None
    prob_file   = find_matching_file(image_id, dirs["pred"], "*.npz")
    if prob_file:
        try:
            data = np.load(prob_file)
            raw  = data[list(data.keys())[0]]
            if raw.size == 2 * np.prod(ref_shape):
                probability = raw.reshape((2,) + ref_shape)
            else:
                print(f"Cannot reshape probability for {image_id}")
        except Exception as e:
            print(f"Warning: probability load failed for {image_id}: {e}")

    # ── uncertainty ────────────────────────────────────────────────────
    uncertainty_data = None
    if dirs.get("uncertainty"):
        try:
            raw_unc      = _load_uncertainty_maps(image_id, dirs["uncertainty"])
            uncertainty_data = _align_uncertainty(raw_unc, ref_shape)
        except Exception as e:
            print(f"Warning: uncertainty load failed for {image_id}: {e}")

    # ── lesion evaluation ──────────────────────────────────────────────
    ev = LesionEval(
        id=image_id,
        pred=sitk.ReadImage(pred_file),
        image=ref_image,
        ground=sitk.ReadImage(lesion_file),
    )

    pred_stats, high_unc = analyze_predicted_lesions(
        pred_pos=ev.pred_lesion_candidates,
        ground_pos=ev.ground_lesion_candidates,
        probs=probability,
        uncertainty_data=uncertainty_data,
        id=image_id,
        ref_image=ref_image,
    )
    ground_stats = analyze_ground_truth_lesions(
        ground_pos=ev.ground_lesion_candidates,
        pred_pos=ev.pred_lesion_candidates,
        probs=probability,
        uncertainty_data=uncertainty_data,
        id=image_id,
        ref_image=ref_image,
        high_uncertainty_lesions=high_unc,
    )
    image_stats = analyze_image(
        ev.ground_lesion_candidates,
        ev.pred_lesion_candidates,
        probability,
        uncertainty_data,
        image_id,
        ref_image,
        high_unc,
    )

    return pred_stats, ground_stats, image_stats


# ── bone-metastasis processing ────────────────────────────────────────────────

def process_bone_mets(lesion_file: str, dirs: dict):
    """
    Build and return a fully populated BoneMetastasis object.
    Returns None if required files are missing.
    """
    image_id = os.path.basename(lesion_file).split(".")[0]

    image_file = find_matching_file(image_id, dirs["images"], "*0.nii.gz")
    bone_file  = find_matching_file(image_id, dirs["bone"],   "*ts.nii.gz")
    pred_file  = find_matching_file(image_id, dirs["pred"],   "*.nii.gz")

    if not (image_file and bone_file and pred_file):
        missing = [n for n, f in [("image", image_file),
                                   ("bone",  bone_file),
                                   ("pred",  pred_file)] if not f]
        print(f"Skipping {image_id} – missing: {', '.join(missing)}")
        return None

    obj = BoneMetastasis(image=image_file, lesion=lesion_file,
                         bone=bone_file,   pred=pred_file)
    obj.lookup_table()
    obj.add_voxel_volume()
    obj.add_priority("ground")
    obj.add_priority("pred")
    obj.add_high_risk()
    obj.merged_table = obj.merge_tables()
    obj.metrics      = obj.calculate_metrics()
    return obj