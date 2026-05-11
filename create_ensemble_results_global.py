import os
import numpy as np
import SimpleITK as sitk
import pickle
from glob import glob

def compute_entropy(prob, axis=0):
    prob = np.clip(prob, epsilon, 1.0)
    return -np.sum(prob * np.log(prob), axis=axis)

# --- Configuration ---
epsilon = 1e-8
mode = 'train'
base_dir = "/dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/"

COMBINED_FOLDER_NAME = "fold_10_ensemble_results"

ens_dirs = [os.path.join(base_dir, f"fold_{i}", mode + '_predictions') for i in range(5)]

print(f"Ensemble directories:\n" + "\n".join(ens_dirs)
      )
combined_dir = os.path.join(base_dir, COMBINED_FOLDER_NAME, mode + '_predictions')
uncer_dir = os.path.join(base_dir, COMBINED_FOLDER_NAME, mode + '_uncertainty_maps')
uncer_raw_dir = os.path.join(base_dir, COMBINED_FOLDER_NAME, mode + '_uncertainty_maps_raw')

os.makedirs(combined_dir, exist_ok=True)
os.makedirs(uncer_dir, exist_ok=True)
os.makedirs(uncer_raw_dir, exist_ok=True)

npz_files = sorted(glob(os.path.join(ens_dirs[1], "*.npz")))
case_ids = [os.path.basename(f).replace(".npz", "") for f in npz_files]
print(f'Processing {len(case_ids)} files')

# =============================================================================
# PASS 1: Compute raw uncertainties, save them, and track global min/max
# =============================================================================
print("\n--- Pass 1: Computing raw uncertainties and global statistics ---")

global_stats = {
    "aleatoric": {"min": np.inf, "max": -np.inf},
    "epistemic": {"min": np.inf, "max": -np.inf},
    "total":     {"min": np.inf, "max": -np.inf},
}

# Optional: also track running stats for percentile-based normalization
# We'll collect per-case percentiles to approximate global percentiles
percentile_tracker = {
    "aleatoric": {"p1_vals": [], "p99_vals": [], "means": [], "stds": []},
    "epistemic": {"p1_vals": [], "p99_vals": [], "means": [], "stds": []},
    "total":     {"p1_vals": [], "p99_vals": [], "means": [], "stds": []},
}

for case_id in case_ids:
    print(f"  [Pass 1] {case_id}")
    prob_list = []
    entropy_list = []

    for ens_dir in ens_dirs:
        npz_path = os.path.join(ens_dir, case_id + ".npz")
        data = np.load(npz_path)
        prob = data["probabilities"]
        prob_list.append(prob)
        member_entropy = compute_entropy(prob, axis=0)
        entropy_list.append(member_entropy)

    avg_prob = np.mean(prob_list, axis=0)

    # Compute raw uncertainties (NO normalization)
    aleatoric_uncertainty = np.mean(entropy_list, axis=0)
    total_uncertainty = compute_entropy(avg_prob, axis=0)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    # Segmentation
    seg = np.argmax(avg_prob, axis=0).astype(np.uint8)
    mask = (seg > 0).astype(np.uint8)
    
    # Save reference info
    ref_nii_path = os.path.join(ens_dirs[0], case_id + ".nii.gz")
    ref_img = sitk.ReadImage(ref_nii_path)

    # Save segmentation
    seg_img = sitk.GetImageFromArray(seg)
    seg_img.CopyInformation(ref_img)
    nii_out_path = os.path.join(combined_dir, case_id + ".nii.gz")
    sitk.WriteImage(seg_img, nii_out_path)

    # Save averaged softmax
    npz_out_path = os.path.join(combined_dir, case_id + ".npz")
    np.savez_compressed(npz_out_path, softmax=avg_prob)

    # Copy properties
    pkl_in_path = os.path.join(ens_dirs[0], case_id + ".pkl")
    pkl_out_path = os.path.join(combined_dir, case_id + ".pkl")
    with open(pkl_in_path, 'rb') as f_in:
        properties = pickle.load(f_in)
    with open(pkl_out_path, 'wb') as f_out:
        pickle.dump(properties, f_out)

    # Save RAW (unnormalized) uncertainty maps and update global stats
    for name, array in zip(
        ["aleatoric", "epistemic", "total"],
        [aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty]
    ):
        # Save raw uncertainty as .nii.gz
        unc_img = sitk.GetImageFromArray(array.astype(np.float32))
        unc_img.CopyInformation(ref_img)
        raw_path = os.path.join(uncer_raw_dir, f"{case_id}_{name}_uncertainty.nii.gz")
        sitk.WriteImage(unc_img, raw_path)

        # Update global min/max
        arr_min, arr_max = float(np.min(array)), float(np.max(array))
        global_stats[name]["min"] = min(global_stats[name]["min"], arr_min)
        global_stats[name]["max"] = max(global_stats[name]["max"], arr_max)

        # Track percentiles for robust normalization
        percentile_tracker[name]["p1_vals"].append(float(np.percentile(array, 1)))
        percentile_tracker[name]["p99_vals"].append(float(np.percentile(array, 99)))
        percentile_tracker[name]["means"].append(float(np.mean(array)))
        percentile_tracker[name]["stds"].append(float(np.std(array)))

# Compute global normalization bounds
# Option A: strict global min/max
# Option B: robust percentile-based (recommended for pseudo-labels)
print("\n--- Global Statistics ---")
global_norm_bounds = {}
for name in ["aleatoric", "epistemic", "total"]:
    gmin = global_stats[name]["min"]
    gmax = global_stats[name]["max"]

    # Robust bounds: use the median of per-case 1st/99th percentiles
    robust_low = float(np.median(percentile_tracker[name]["p1_vals"]))
    robust_high = float(np.percentile(percentile_tracker[name]["p99_vals"], 95))
    # ^ use 95th percentile of per-case p99 values to be robust to outlier cases

    print(f"  {name}: global_min={gmin:.6f}, global_max={gmax:.6f}, "
          f"robust_low={robust_low:.6f}, robust_high={robust_high:.6f}")

    # Choose which bounds to use (uncomment your preference):

    # Option A: Global min/max (preserves full range)
    global_norm_bounds[name] = (gmin, gmax)
 
# Save global stats for reproducibility
stats_path = os.path.join(uncer_dir, "global_uncertainty_stats.pkl")
with open(stats_path, 'wb') as f:
    pickle.dump({
        "global_stats": global_stats,
        "percentile_tracker": percentile_tracker,
        "global_norm_bounds": global_norm_bounds
    }, f)
print(f"  Saved global stats to {stats_path}")

# =============================================================================
# PASS 2: Reload raw uncertainties and normalize globally
# =============================================================================
print("\n--- Pass 2: Globally normalizing uncertainty maps ---")

for case_id in case_ids:
    print(f"  [Pass 2] {case_id}")

    # Load reference for CopyInformation
    ref_nii_path = os.path.join(ens_dirs[0], case_id + ".nii.gz")
    ref_img = sitk.ReadImage(ref_nii_path)

    for name in ["aleatoric", "epistemic", "total"]:
        # Load raw uncertainty
        raw_path = os.path.join(uncer_raw_dir, f"{case_id}_{name}_uncertainty.nii.gz")
        raw_img = sitk.ReadImage(raw_path)
        array = sitk.GetArrayFromImage(raw_img).astype(np.float32)

        # Global normalization
        low, high = global_norm_bounds[name]
        if high > low:
            arr_norm = (array - low) / (high - low)
            arr_norm = np.clip(arr_norm, 0.0, 1.0)  # clip to [0, 1]
        else:
            arr_norm = np.zeros_like(array)

        # Save globally normalized uncertainty
        unc_img = sitk.GetImageFromArray(arr_norm.astype(np.float32))
        unc_img.CopyInformation(ref_img)
        out_path = os.path.join(uncer_dir, f"{case_id}_{name}_uncertainty.nii.gz")
        sitk.WriteImage(unc_img, out_path)

print(f"\n✅ Done! Globally normalized uncertainty maps in:\n{uncer_dir}")
print(f"   Raw (unnormalized) maps preserved in:\n{uncer_raw_dir}")