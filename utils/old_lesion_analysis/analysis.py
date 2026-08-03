import numpy as np
import SimpleITK as sitk
import pandas as pd

"""
This module contains functions for analyzing segmentation results at both lesion and image levels.
It computes various statistics including true positives, false negatives, probabilities, and uncertainties.
"""


# CURRENTLY FILTERS BASED ON UNCERTAINTY LEVEL

################################ LESION LEVEL ######################################

def analyze_ground_truth_lesions(ground_pos, pred_pos, probs=None, uncertainty_data=None, id=None, ref_image=None, high_uncertainty_lesions=None):
    """Analyze ground truth lesions and extract statistics
    Get the probability and uncertaintites for ground truth lesions. 
    Primary use is to calculate the number of false negatives. 

    Args:
        ground_pos (array): Array of ground truth lesion positions
        pred_pos (array): Array of predicted lesion positions
        probs (array, optional): Array of probabilities for each class. Can be None.
        uncertainty_data (dict, optional): Dictionary containing uncertainty data. Can be None.
        id (str, optional): Image identifier
            ref_image (SimpleITK.Image, optional): Reference image for spatial information
        high_uncertainty_lesions (DataFrame, optional): High uncertainty lesions to filter out

    Returns:
        DataFrame: Ground truth lesion statistics
    """

    OVERLAP_THRESH = 1 # If overlap percentage is above this number, then it's a True Positive

    # Handle high_uncertainty_lesions safely
    if high_uncertainty_lesions is not None and len(high_uncertainty_lesions) > 0:
        if 'pred_lesion_id' in high_uncertainty_lesions.columns:
            for lesion in high_uncertainty_lesions['pred_lesion_id']:
                if pred_pos is not None:
                    pred_pos[pred_pos == lesion] = 0

    print(f"OVERLAP THRESHOLD: {OVERLAP_THRESH}")
    ground_lesion_stats = []
    image_mask = np.zeros_like(ground_pos, dtype=bool) 

    for lesion in np.unique(ground_pos):
        if lesion == 0:
            continue
        image_mask[ground_pos == lesion] = True
        # Create mask for this specific ground truth lesion
        lesion_mask = np.zeros_like(ground_pos, dtype=bool)
        lesion_mask[ground_pos == lesion] = True

        # Get probabilities for both classes at this lesion location - handle None case
        if probs is not None and len(probs) >= 2:
            lesion_probs_class0 = probs[0][lesion_mask]
            lesion_probs_class1 = probs[1][lesion_mask]
            
            # Calculate average probabilities
            avg_prob_class0 = np.mean(lesion_probs_class0)
            avg_prob_class1 = np.mean(lesion_probs_class1)
        else:
            avg_prob_class0 = None
            avg_prob_class1 = None
        
        # Calculate uncertainty metrics - handle None case
        if uncertainty_data is not None:
            avg_epistemic = np.mean(uncertainty_data['epistemic'][lesion_mask]) if uncertainty_data.get('epistemic') is not None else None
            avg_aleatoric = np.mean(uncertainty_data['aleatoric'][lesion_mask]) if uncertainty_data.get('aleatoric') is not None else None
            avg_total = np.mean(uncertainty_data['total'][lesion_mask]) if uncertainty_data.get('total') is not None else None
        else:
            avg_epistemic = None
            avg_aleatoric = None
            avg_total = None

        # Count overlap with predictions
        overlap_count_with_predicted = 0
        if pred_pos is not None:
            overlap_count_with_predicted = np.sum(pred_pos[lesion_mask] > 0)

        total_gt_pixels = np.sum(lesion_mask)
        overlap_percentage = overlap_count_with_predicted / total_gt_pixels * 100 if total_gt_pixels > 0 else 0

        TP_pixels = int(overlap_count_with_predicted)
        FN_pixels = int(total_gt_pixels - overlap_count_with_predicted)

        # Create a mask over the TP_pixels and the FN_pixels to get the uncertainty for those
        tp_mask = np.zeros_like(lesion_mask, dtype=bool)
        fn_mask = np.zeros_like(lesion_mask, dtype=bool)

        if pred_pos is not None:
            tp_mask[lesion_mask] = (pred_pos[lesion_mask] > 0)
            fn_mask[lesion_mask] = (pred_pos[lesion_mask] == 0)

        # Handle uncertainty and probability data safely
        if uncertainty_data is not None and uncertainty_data.get('total') is not None:
            tp_uncertainty = uncertainty_data['total'][tp_mask] if np.any(tp_mask) else None
            fn_uncertainty = uncertainty_data['total'][fn_mask] if np.any(fn_mask) else None
        else:
            tp_uncertainty = None
            fn_uncertainty = None

        if probs is not None and len(probs) >= 2:
            tp_prob_class0 = probs[0][tp_mask] if np.any(tp_mask) else None
            tp_prob_class1 = probs[1][tp_mask] if np.any(tp_mask) else None
            fn_prob_class0 = probs[0][fn_mask] if np.any(fn_mask) else None
            fn_prob_class1 = probs[1][fn_mask] if np.any(fn_mask) else None
        else:
            tp_prob_class0 = None
            tp_prob_class1 = None
            fn_prob_class0 = None
            fn_prob_class1 = None

        pred_lesion_id = None  # ensure defined even if overlap threshold not met
        if (TP_pixels/total_gt_pixels)*100 > OVERLAP_THRESH and pred_pos is not None:
            pred_lesion_id = np.unique(pred_pos[lesion_mask])

        ############### VOLUME ##################
        spacing = ref_image.GetSpacing() if isinstance(ref_image, sitk.Image) else (1.0, 1.0, 1.0)
        volume_per_pixel = np.prod(spacing)

        # Store statistics
        ground_lesion_stats.append({
            # Meta Data
            'image_id': id,
            'gt_lesion_id': lesion,
            'pred_lesion_id': pred_lesion_id,
            'lesion_type': 'ground_truth',

            # Pixel Info
            'total_pixels_gt': total_gt_pixels,
            'TP_pixels_gt': overlap_count_with_predicted,
            'TP_pixels_volume_gt': TP_pixels * volume_per_pixel,
            'TP_pixels_uncertainty_gt': tp_uncertainty,
            'TP_prob_lesion_gt': tp_prob_class1,
            'TP_prob_background_gt': tp_prob_class0,

            'FN_pixels_gt': FN_pixels,
            'FN_pixels_volume_gt': FN_pixels * volume_per_pixel,
            'FN_uncertainty_gt': fn_uncertainty,
            'FN_prob_lesion_gt': fn_prob_class1,
            'FN_prob_background_gt': fn_prob_class0,

            # Volume info
            'gt_lesion_volume': total_gt_pixels * volume_per_pixel,

            # Percentage count
            'overlap_percentage_gt_with_predicted': overlap_percentage,

            # Probability and uncertainty info
            'avg_prob_background_gt': avg_prob_class0,
            'avg_prob_lesion_gt': avg_prob_class1, 

            'avg_epistemic_uncertainty_gt': avg_epistemic,
            'avg_aleatoric_uncertainty_gt': avg_aleatoric,
            'avg_total_uncertainty_gt': avg_total,
            'Detection': 'TP' if overlap_percentage > OVERLAP_THRESH else 'FN'
        })
    # Description of all the stats (ground_lesion_stats entries)
    # - image_id: String identifier of the image/case.
    # - gt_lesion_id: Integer label ID of the ground-truth lesion in the connected-component map.
    # - pred_lesion_id: List of overlapping predicted lesion IDs (excluding background) when overlap > 1%; None otherwise.
    # - lesion_type: Always 'ground_truth' for entries produced by this function.
    # - total_pixels_gt: Total voxel count for this GT lesion.
    # - TP_pixels_gt: Voxel count in this GT lesion that overlaps any predicted lesion (true positives at voxel level).
    # - FN_pixels_gt: Voxel count in this GT lesion not overlapped by prediction (false negatives at voxel level).
    # - gt_lesion_volume: Volume (mm^3) of the GT lesion (total_pixels_gt × voxel volume).
    # - TP_volume_gt: Overlap volume (mm^3) between GT and prediction (TP_pixels_gt × voxel volume).
    # - FN_volume_gt: Missed volume (mm^3) in GT not predicted (FN_pixels_gt × voxel volume).
    # - overlap_percentage_gt_with_predicted: TP_pixels_gt / total_pixels_gt × 100 (0 if denominator is 0).
    # - avg_prob_background_gt: Mean P(class=0) across voxels of this GT lesion.
    # - avg_prob_lesion_gt: Mean P(class=1) across voxels of this GT lesion.
    # - avg_epistemic_uncertainty_gt: Mean epistemic uncertainty across voxels of this GT lesion (None if unavailable).
    # - avg_aleatoric_uncertainty_gt: Mean aleatoric uncertainty across voxels of this GT lesion (None if unavailable).
    # - avg_total_uncertainty_gt: Mean total uncertainty across voxels of this GT lesion (None if unavailable).

    return pd.DataFrame(ground_lesion_stats)

def analyze_predicted_lesions(pred_pos, ground_pos=None, probs=None, uncertainty_data=None, id=None, ref_image=None, UNCERTAINTY_CUTOFF=None):
    """Analyze predicted lesions and extract statistics"""
    pred_lesion_stats = []
    high_uncertainty_lesions = []

    if UNCERTAINTY_CUTOFF is not None:
        print("NOTICE: RUNNING WITH UNCERTAINTY CUTOFF =", UNCERTAINTY_CUTOFF)

    for lesion in np.unique(pred_pos):
        if lesion == 0:
            continue
        
        # Create mask for this specific predicted lesion
        lesion_mask = np.zeros_like(pred_pos, dtype=bool)
        lesion_mask[pred_pos == lesion] = True
        
        # Get probabilities for both classes - handle None case
        if probs is not None and len(probs) >= 2:
            lesion_probs_class0 = probs[0][lesion_mask]
            lesion_probs_class1 = probs[1][lesion_mask]
            
            # Calculate average probabilities (use _pred naming)
            avg_prob_pred_background = np.mean(lesion_probs_class0)
            avg_prob_pred_lesion = np.mean(lesion_probs_class1)
        else:
            avg_prob_pred_background = None
            avg_prob_pred_lesion = None

        # Calculate uncertainty metrics (use _pred naming)
        if uncertainty_data is not None:
            avg_epistemic_uncertainty_pred = (
                np.mean(uncertainty_data['epistemic'][lesion_mask])
                if uncertainty_data.get('epistemic') is not None else None
            )
            avg_aleatoric_uncertainty_pred = (
                np.mean(uncertainty_data['aleatoric'][lesion_mask])
                if uncertainty_data.get('aleatoric') is not None else None
            )
            avg_total_uncertainty_pred = (
                np.mean(uncertainty_data['total'][lesion_mask])
                if uncertainty_data.get('total') is not None else None
            )
        else:
            avg_epistemic_uncertainty_pred = None
            avg_aleatoric_uncertainty_pred = None
            avg_total_uncertainty_pred = None

        # Count overlap with ground truth
        overlap_count_with_ground = 0
        if ground_pos is not None:
            overlap_count_with_ground = np.sum(ground_pos[lesion_mask] > 0)

        total_pred_pixels = int(np.sum(lesion_mask))
        overlap_percentage = overlap_count_with_ground / total_pred_pixels * 100 if total_pred_pixels > 0 else 0
        TP_pixels = int(overlap_count_with_ground)
        FP_pixels = int(total_pred_pixels - overlap_count_with_ground)

        fp_mask = np.zeros_like(lesion_mask, dtype=bool)
        if ground_pos is not None:
            fp_mask[lesion_mask] = (ground_pos[lesion_mask] == 0)
        
        # Handle uncertainty and probability data for FP pixels
        if uncertainty_data is not None and uncertainty_data.get('total') is not None:
            fp_uncertainty = uncertainty_data['total'][fp_mask] if np.any(fp_mask) else None
        else:
            fp_uncertainty = None
            
        if probs is not None and len(probs) >= 2:
            fp_prob_class0 = probs[0][fp_mask] if np.any(fp_mask) else None
            fp_prob_class1 = probs[1][fp_mask] if np.any(fp_mask) else None
        else:
            fp_prob_class0 = None
            fp_prob_class1 = None

        high_uncertainty_lesions = pd.DataFrame()

        gt_lesion_id = None
        # if the overlap is over 1%, also add the lesion_id for ground
        if (TP_pixels/total_pred_pixels)*100 > 1 and ground_pos is not None: 
            gt_lesion_id = np.unique(ground_pos[lesion_mask])

        ############### VOLUME ##################
        spacing = ref_image.GetSpacing() if isinstance(ref_image, sitk.Image) else (1.0, 1.0, 1.0)
        volume_per_pixel = np.prod(spacing)

        # Store statistics (mirror GT naming scheme)
        pred_lesion_stats.append({
            'image_id': id,
            'pred_lesion_id': lesion,
            'gt_lesion_id': gt_lesion_id,
            'lesion_type': 'predicted',

            # Pixel Info
            'total_pixels_pred': total_pred_pixels,
            'TP_pixels_pred': TP_pixels,
            'TP_pixels_volume_pred': TP_pixels * volume_per_pixel,

            'FP_pixels_pred': FP_pixels,
            'FP_pixels_volume_pred': FP_pixels * volume_per_pixel,
            'FP_uncertainty_pred': fp_uncertainty,
            'FP_prob_background_pred': fp_prob_class0,
            'FP_prob_lesion_pred': fp_prob_class1,

            # Volume info
            'pred_lesion_volume': total_pred_pixels * volume_per_pixel,

            # Percentage colesion
            'overlap_percentage_pred_with_gt': overlap_percentage,
            # Probabilities and Uncertainty
            'avg_prob_background_pred': avg_prob_pred_background,
            'avg_prob_pred_lesion': avg_prob_pred_lesion,
            'avg_epistemic_uncertainty_pred': avg_epistemic_uncertainty_pred,

            'avg_aleatoric_uncertainty_pred': avg_aleatoric_uncertainty_pred,
            'avg_total_uncertainty_pred': avg_total_uncertainty_pred,

            'Detection': 'TP' if overlap_percentage > 1 else 'FP', 
        })
        
    return pd.DataFrame(pred_lesion_stats), pd.DataFrame(high_uncertainty_lesions)

def save_lesion_masks(ground_pos, pred_pos, ref_image, id, LESION_SAVE_FOLDER_DIR):
    """Save lesion masks as NIfTI files"""
    # Save ground truth lesions
    gt_lesion_mask = np.zeros_like(ground_pos, dtype=np.uint8)
    for lesion in np.unique(ground_pos):
        if lesion == 0:
            continue
        gt_lesion_mask[ground_pos == lesion] = lesion

    gt_lesion_sitk = sitk.GetImageFromArray(gt_lesion_mask)
    gt_lesion_sitk.CopyInformation(ref_image)
    sitk.WriteImage(gt_lesion_sitk, f"./{LESION_SAVE_FOLDER_DIR}/{id}_gt_all_lesions.nii.gz")

    # Save predicted lesions
    pred_lesion_mask = np.zeros_like(pred_pos, dtype=np.uint8)
    for lesion in np.unique(pred_pos):
        if lesion == 0:
            continue
        pred_lesion_mask[pred_pos == lesion] = lesion
    
    pred_lesion_sitk = sitk.GetImageFromArray(pred_lesion_mask)
    pred_lesion_sitk.CopyInformation(ref_image)
    sitk.WriteImage(pred_lesion_sitk, f"./{LESION_SAVE_FOLDER_DIR}/{id}_pred_all_lesions.nii.gz")


###############################  IMAGE LEVEL  ######################################

def analyze_image(ground_pos, pred_pos, probability_data=None, uncertainty_data=None, id=None, ref_image=None, high_uncertainty_lesions=None):
    """Compute image-level voxel and lesion metrics.

    Args:
        ground_pos (ndarray): Labeled GT mask (0=background, >0 lesion ids)
        pred_pos (ndarray): Labeled prediction mask (0=background, >0 lesion ids)
        probability_data (ndarray, optional): Softmax probs with shape (2, Z, Y, X) or (2, Y, X, Z) or (2, H, W, D)
        uncertainty_data (dict, optional): {'aleatoric': ndarray|None, 'epistemic': ndarray|None, 'total': ndarray|None}
        id (str, optional): Image identifier
        ref_image (SimpleITK.Image, optional): For spacing (voxel volume)
        high_uncertainty_lesions (DataFrame, optional): High uncertainty lesions to filter out
    
    Returns:
        dict: Image-level metrics summary
    """


    # Ensure numpy arrays
    gt = np.asarray(ground_pos)
    pred = np.asarray(pred_pos)

    # remove high uncertainty lesions from predicted image
    if high_uncertainty_lesions is not None and len(high_uncertainty_lesions) > 0:
        if 'pred_lesion_id' in high_uncertainty_lesions.columns:
            for lesion in high_uncertainty_lesions['pred_lesion_id']:
                pred[pred == lesion] = 0
        
    # Binary masks
    gt_bin = (gt > 0)
    pred_bin = (pred > 0)

    # Spacing and voxel volume
    spacing = ref_image.GetSpacing() if isinstance(ref_image, sitk.Image) else (1.0, 1.0, 1.0)
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])

    # Voxel-level confusion
    TP_vox = int(np.sum(gt_bin & pred_bin))
    FN_vox = int(np.sum(gt_bin & ~pred_bin))
    FP_vox = int(np.sum(~gt_bin & pred_bin))
    TN_vox = int(np.sum(~gt_bin & ~pred_bin))

    FN_vol = float(FN_vox * voxel_volume)
    FP_vol = float(FP_vox * voxel_volume)

    denom_dice = (2 * TP_vox + FN_vox + FP_vox)
    dice = float((2 * TP_vox) / denom_dice) if denom_dice > 0 else 0.0

    sens_voxel = float(TP_vox / (TP_vox + FN_vox)) if (TP_vox + FN_vox) > 0 else 0.0
    ppv_voxel = float(TP_vox / (TP_vox + FP_vox)) if (TP_vox + FP_vox) > 0 else 0.0

    # Lesion-level metrics
    gt_labels = np.unique(gt)
    gt_labels = gt_labels[gt_labels > 0]
    pred_labels = np.unique(pred)
    pred_labels = pred_labels[pred_labels > 0]

    # Initialize uncertainty and probability variables
    bg_unc_ale = bg_unc_tot = bg_unc_epi = None
    les_unc_ale = les_unc_tot = les_unc_epi = None
    bg_prob_bg = bg_prob_les = les_prob_bg = les_prob_les = None

    # Slice-aware background mask (slices that contain GT lesions)
    if gt_bin.ndim == 3:
        slices = np.any(gt_bin, axis=(0, 1))
        slice_mask = slices[np.newaxis, np.newaxis, :]
        background_mask = (~gt_bin) & slice_mask
    else:
        # Fallback: use all non-lesion voxels
        background_mask = ~gt_bin

    lesion_mask = gt_bin

    # Uncertainty summaries - handle None case
    if uncertainty_data is not None:
        if uncertainty_data.get('aleatoric') is not None:
            les_unc_ale = float(np.mean(uncertainty_data['aleatoric'][lesion_mask])) if np.any(lesion_mask) else None
            bg_unc_ale = float(np.mean(uncertainty_data['aleatoric'][background_mask])) if np.any(background_mask) else None
        if uncertainty_data.get('epistemic') is not None:
            les_unc_epi = float(np.mean(uncertainty_data['epistemic'][lesion_mask])) if np.any(lesion_mask) else None
            bg_unc_epi = float(np.mean(uncertainty_data['epistemic'][background_mask])) if np.any(background_mask) else None
        if uncertainty_data.get('total') is not None:
            les_unc_tot = float(np.mean(uncertainty_data['total'][lesion_mask])) if np.any(lesion_mask) else None
            bg_unc_tot = float(np.mean(uncertainty_data['total'][background_mask])) if np.any(background_mask) else None

    # Probability summaries (handle possible channel-first probabilities) - handle None case
    if probability_data is not None and isinstance(probability_data, np.ndarray) and probability_data.ndim >= 3:
        probs = probability_data
        # Ensure channel-first: (2, ..., ..., ...)
        if probs.shape[0] != 2 and probs.shape[-1] == 2:
            # Move last channel to first if needed
            probs = np.moveaxis(probs, -1, 0)
    
        if probs.shape[0] == 2:
            p_bg = probs[0]
            p_les = probs[1]
            les_prob_bg = float(np.mean(p_bg[lesion_mask])) if np.any(lesion_mask) else None
            les_prob_les = float(np.mean(p_les[lesion_mask])) if np.any(lesion_mask) else None
            bg_prob_bg = float(np.mean(p_bg[background_mask])) if np.any(background_mask) else None
            bg_prob_les = float(np.mean(p_les[background_mask])) if np.any(background_mask) else None

    # Bundle image-level metrics
    image_metrics = {
        # Metadata
        'image_id': id,

        # Voxel-level
        'TP_voxels': TP_vox,
        'FN_voxels': FN_vox,
        'FP_voxels': FP_vox,
        'TN_voxels': TN_vox,

        # Volumes
        'FN_volume_mm3': FN_vol,
        'FP_volume_mm3': FP_vol,
        'TP_volume_mm3': float(TP_vox * voxel_volume),
        
        'Dice_coefficient': dice,
        'Sensitivity_voxel': sens_voxel,
        'PPV_voxel': ppv_voxel,

        # Lesion-level
        'GT_lesions': int(len(gt_labels)),
        'Predicted_lesions': int(len(pred_labels)),
 
        # Uncertainty summaries (lesion vs slice-aware background)
        'Avg_Lesion_Aleatoric_Unc': les_unc_ale,
        'Avg_Lesion_Epistemic_Unc': les_unc_epi,
        'Avg_Lesion_Total_Unc': les_unc_tot,

        'Avg_Background_Aleatoric_Unc': bg_unc_ale,
        'Avg_Background_Epistemic_Unc': bg_unc_epi,
        'Avg_Background_Total_Unc': bg_unc_tot,

        # Probability summaries
        'Avg_Lesion_Prob_Background': les_prob_bg,
        'Avg_Lesion_Prob_Lesion': les_prob_les,
        'Avg_Background_Prob_Background': bg_prob_bg,
        'Avg_Background_Prob_Lesion': bg_prob_les,
    }

    return image_metrics

