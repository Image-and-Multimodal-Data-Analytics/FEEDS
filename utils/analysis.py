# analysis.py
"""
Vectorised lesion- and image-level analysis functions.
All loops are kept to a minimum; numpy boolean indexing is used throughout.
"""

import numpy as np
import SimpleITK as sitk
import pandas as pd
import nibabel as nib
import cc3d

OVERLAP_THRESH = 1   # percent


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _safe_mean(arr, mask):
    """Mean of arr[mask], or None if mask is empty."""
    return float(np.mean(arr[mask])) if arr is not None and np.any(mask) else None


def _voxel_volume(ref_image):
    if isinstance(ref_image, sitk.Image):
        return float(np.prod(ref_image.GetSpacing()))
    return 1.0


# ── predicted lesions ─────────────────────────────────────────────────────────
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

def analyze_predicted_lesions(pred_pos, ground_pos=None, probs=None,
                              uncertainty_data=None, id=None,
                              ref_image=None, UNCERTAINTY_CUTOFF=None):
    """
    Returns
    -------
    pred_df : pd.DataFrame   – one row per predicted lesion
    high_unc_df : pd.DataFrame – subset with high uncertainty
    """
    vv   = _voxel_volume(ref_image)
    rows = []
    high_unc_rows = []

    # pre-extract probability / uncertainty arrays once
    p0   = probs[0]            if (probs is not None and len(probs) >= 2) else None
    p1   = probs[1]            if (probs is not None and len(probs) >= 2) else None
    epi  = uncertainty_data.get("epistemic") if uncertainty_data else None
    ale  = uncertainty_data.get("aleatoric") if uncertainty_data else None
    tot  = uncertainty_data.get("total")     if uncertainty_data else None

    pred_labels = np.unique(pred_pos)
    pred_labels = pred_labels[pred_labels > 0]

    for lesion in pred_labels:
        lmask = pred_pos == lesion   # bool mask for this predicted lesion

        total_pred_px = int(lmask.sum())

        # overlap with GT
        TP_px = int((lmask & (ground_pos > 0)).sum()) if ground_pos is not None else 0
        FP_px = total_pred_px - TP_px
        overlap_pct = TP_px / total_pred_px * 100 if total_pred_px > 0 else 0

        gt_lesion_id = (
            np.unique(ground_pos[lmask & (ground_pos > 0)]).tolist()
            if (ground_pos is not None and overlap_pct > OVERLAP_THRESH) else None
        )

        row = {
            "image_id":               id,
            "pred_lesion_id":         lesion,
            "gt_lesion_id":           gt_lesion_id,
            "lesion_type":            "predicted",
            "total_pixels_pred":      total_pred_px,
            "TP_pixels_pred":         TP_px,
            "TP_pixels_volume_pred":  TP_px * vv,
            "FP_pixels_pred":         FP_px,
            "FP_pixels_volume_pred":  FP_px * vv,
            "pred_lesion_volume":     total_pred_px * vv,
            "overlap_percentage_pred_with_gt": overlap_pct,
            # probabilities
            "avg_prob_background_pred":  _safe_mean(p0,  lmask),
            "avg_prob_pred_lesion":      _safe_mean(p1,  lmask),
            # uncertainty
            "avg_epistemic_uncertainty_pred": _safe_mean(epi, lmask),
            "avg_aleatoric_uncertainty_pred": _safe_mean(ale, lmask),
            "avg_total_uncertainty_pred":     _safe_mean(tot, lmask),
            # FP uncertainty / prob (voxels that are FP)
            "FP_uncertainty_pred":    tot[lmask & (ground_pos == 0)].tolist()
                                      if (tot is not None and ground_pos is not None) else None,
            "FP_prob_background_pred": _safe_mean(p0, lmask & (ground_pos == 0))
                                       if ground_pos is not None else None,
            "FP_prob_lesion_pred":     _safe_mean(p1, lmask & (ground_pos == 0))
                                       if ground_pos is not None else None,
            "Detection": "TP" if overlap_pct > OVERLAP_THRESH else "FP",
        }

        rows.append(row)

        # high-uncertainty tracking
        if UNCERTAINTY_CUTOFF is not None and tot is not None:
            if _safe_mean(tot, lmask) is not None and _safe_mean(tot, lmask) > UNCERTAINTY_CUTOFF:
                high_unc_rows.append({"pred_lesion_id": lesion,
                                      "avg_total_uncertainty": _safe_mean(tot, lmask)})

    return pd.DataFrame(rows), pd.DataFrame(high_unc_rows)


# ── ground-truth lesions ──────────────────────────────────────────────────────

def analyze_ground_truth_lesions(ground_pos, pred_pos, probs=None,
                                 uncertainty_data=None, id=None,
                                 ref_image=None, high_uncertainty_lesions=None):
    """Returns a DataFrame with one row per GT lesion."""
    vv = _voxel_volume(ref_image)

    # optionally remove high-uncertainty predicted lesions
    if (high_uncertainty_lesions is not None
            and len(high_uncertainty_lesions) > 0
            and "pred_lesion_id" in high_uncertainty_lesions.columns):
        pred_pos = pred_pos.copy()
        for pid in high_uncertainty_lesions["pred_lesion_id"]:
            pred_pos[pred_pos == pid] = 0

    p0  = probs[0]            if (probs is not None and len(probs) >= 2) else None
    p1  = probs[1]            if (probs is not None and len(probs) >= 2) else None
    epi = uncertainty_data.get("epistemic") if uncertainty_data else None
    ale = uncertainty_data.get("aleatoric") if uncertainty_data else None
    tot = uncertainty_data.get("total")     if uncertainty_data else None

    gt_labels = np.unique(ground_pos)
    gt_labels = gt_labels[gt_labels > 0]

    rows = []
    for lesion in gt_labels:
        lmask = ground_pos == lesion
        total_gt_px  = int(lmask.sum())

        TP_px  = int((lmask & (pred_pos > 0)).sum())
        FN_px  = total_gt_px - TP_px
        overlap_pct = TP_px / total_gt_px * 100 if total_gt_px > 0 else 0

        tp_mask = lmask & (pred_pos > 0)
        fn_mask = lmask & (pred_pos == 0)

        pred_lesion_id = (
            np.unique(pred_pos[tp_mask]).tolist()
            if (overlap_pct > OVERLAP_THRESH and pred_pos is not None) else None
        )

        rows.append({
            "image_id":      id,
            "gt_lesion_id":  lesion,
            "pred_lesion_id": pred_lesion_id,
            "lesion_type":   "ground_truth",
            # pixel counts
            "total_pixels_gt":       total_gt_px,
            "TP_pixels_gt":          TP_px,
            "TP_pixels_volume_gt":   TP_px * vv,
            "FN_pixels_gt":          FN_px,
            "FN_pixels_volume_gt":   FN_px * vv,
            "gt_lesion_volume":      total_gt_px * vv,
            "overlap_percentage_gt_with_predicted": overlap_pct,
            # TP masks for uncertainty / prob
            "TP_pixels_uncertainty_gt": _safe_mean(tot, tp_mask),
            "TP_prob_lesion_gt":        _safe_mean(p1,  tp_mask),
            "TP_prob_background_gt":    _safe_mean(p0,  tp_mask),
            "FN_uncertainty_gt":        _safe_mean(tot, fn_mask),
            "FN_prob_lesion_gt":        _safe_mean(p1,  fn_mask),
            "FN_prob_background_gt":    _safe_mean(p0,  fn_mask),
            # per-lesion averages
            "avg_prob_background_gt":           _safe_mean(p0,  lmask),
            "avg_prob_lesion_gt":               _safe_mean(p1,  lmask),
            "avg_epistemic_uncertainty_gt":     _safe_mean(epi, lmask),
            "avg_aleatoric_uncertainty_gt":     _safe_mean(ale, lmask),
            "avg_total_uncertainty_gt":         _safe_mean(tot, lmask),
            "Detection": "TP" if overlap_pct > OVERLAP_THRESH else "FN",
        })

    return pd.DataFrame(rows)


# ── image level ───────────────────────────────────────────────────────────────

def analyze_image(ground_pos, pred_pos, probability_data=None,
                  uncertainty_data=None, id=None,
                  ref_image=None, high_uncertainty_lesions=None):
    """Compute image-level voxel and lesion metrics."""
    vv = _voxel_volume(ref_image)

    # remove high-uncertainty lesions from pred
    if (high_uncertainty_lesions is not None
            and len(high_uncertainty_lesions) > 0
            and "pred_lesion_id" in high_uncertainty_lesions.columns):
        pred_pos = pred_pos.copy()
        for pid in high_uncertainty_lesions["pred_lesion_id"]:
            pred_pos[pred_pos == pid] = 0

    gt_bin   = ground_pos > 0
    pred_bin = pred_pos   > 0

    TP_vox = int((gt_bin  &  pred_bin).sum())
    FN_vox = int((gt_bin  & ~pred_bin).sum())
    FP_vox = int((~gt_bin &  pred_bin).sum())
    TN_vox = int((~gt_bin & ~pred_bin).sum())

    dice       = 2*TP_vox / (2*TP_vox + FP_vox + FN_vox) if (2*TP_vox+FP_vox+FN_vox) else 0
    sens_voxel = TP_vox / (TP_vox + FN_vox) if (TP_vox + FN_vox) else 0
    ppv_voxel  = TP_vox / (TP_vox + FP_vox) if (TP_vox + FP_vox) else 0

    gt_labels   = np.unique(ground_pos); gt_labels   = gt_labels[gt_labels   > 0]
    pred_labels = np.unique(pred_pos);   pred_labels = pred_labels[pred_labels > 0]

    # slice-aware background mask
    if gt_bin.ndim == 3:
        slices     = np.any(gt_bin, axis=(0, 1))
        slice_mask = slices[np.newaxis, np.newaxis, :]
        bg_mask    = (~gt_bin) & slice_mask
    else:
        bg_mask = ~gt_bin

    lesion_mask = gt_bin   # reuse

    # probabilities
    les_prob_bg = les_prob_les = bg_prob_bg = bg_prob_les = None
    if (probability_data is not None
            and isinstance(probability_data, np.ndarray)
            and probability_data.ndim >= 3):
        probs = probability_data
        if probs.shape[-1] == 2:          # channel-last → channel-first
            probs = np.moveaxis(probs, -1, 0)
        if probs.shape[0] == 2:
            p_bg  = probs[0]; p_les = probs[1]
            les_prob_bg  = _safe_mean(p_bg,  lesion_mask)
            les_prob_les = _safe_mean(p_les, lesion_mask)
            bg_prob_bg   = _safe_mean(p_bg,  bg_mask)
            bg_prob_les  = _safe_mean(p_les, bg_mask)

    # uncertainties
    les_unc_ale = les_unc_epi = les_unc_tot = None
    bg_unc_ale  = bg_unc_epi  = bg_unc_tot  = None
    if uncertainty_data is not None:
        for key, les_var, bg_var in [
            ("aleatoric", "les_unc_ale", "bg_unc_ale"),
            ("epistemic", "les_unc_epi", "bg_unc_epi"),
            ("total",     "les_unc_tot", "bg_unc_tot"),
        ]:
            arr = uncertainty_data.get(key)
            if arr is not None:
                locals()[les_var] = _safe_mean(arr, lesion_mask)
                locals()[bg_var]  = _safe_mean(arr, bg_mask)

    return {
        "image_id": id,
        # lesion counts
        "GT_lesions":        int(len(gt_labels)),
        "Predicted_lesions": int(len(pred_labels)),
        # voxel counts
        "TP_voxels": TP_vox, "FN_voxels": FN_vox,
        "FP_voxels": FP_vox, "TN_voxels": TN_vox,
        # volumes
        "TP_volume_mm3": TP_vox * vv,
        "FN_volume_mm3": FN_vox * vv,
        "FP_volume_mm3": FP_vox * vv,
        # metrics
        "Dice_coefficient":   dice,
        "Sensitivity_voxel":  sens_voxel,
        "PPV_voxel":          ppv_voxel,
        # probabilities
        "Avg_Lesion_Prob_Background":   les_prob_bg,
        "Avg_Lesion_Prob_Lesion":       les_prob_les,
        "Avg_Background_Prob_Background": bg_prob_bg,
        "Avg_Background_Prob_Lesion":   bg_prob_les,
        # uncertainties
        "Avg_Lesion_Aleatoric_Unc":    les_unc_ale,
        "Avg_Lesion_Epistemic_Unc":    les_unc_epi,
        "Avg_Lesion_Total_Unc":        les_unc_tot,
        "Avg_Background_Aleatoric_Unc": bg_unc_ale,
        "Avg_Background_Epistemic_Unc": bg_unc_epi,
        "Avg_Background_Total_Unc":    bg_unc_tot,
    }


# ── save helpers ──────────────────────────────────────────────────────────────

def save_lesion_masks(ground_pos, pred_pos, ref_image, id, save_dir):
    import os
    os.makedirs(save_dir, exist_ok=True)

    def _write(arr, suffix):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(ref_image)
        sitk.WriteImage(img, os.path.join(save_dir, f"{id}_{suffix}.nii.gz"))

    gt_mask   = np.where(ground_pos > 0, ground_pos, 0).astype(np.uint8)
    pred_mask = np.where(pred_pos   > 0, pred_pos,   0).astype(np.uint8)
    _write(gt_mask,   "gt_all_lesions")
    _write(pred_mask, "pred_all_lesions")