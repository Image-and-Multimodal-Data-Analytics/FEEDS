import os
import numpy as np
import SimpleITK as sitk
import pickle
from glob import glob

# --- Configuration ---
epsilon = 1e-8
mode = 'train'
base_dir = "//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_0_10/"

COMBINED_FOLDER_NAME = "train_tta_predicted_bad_subset"

# --- Input: folder containing existing raw uncertainty maps ---
uncer_raw_dir = os.path.join(base_dir, COMBINED_FOLDER_NAME, 'uncertainty_maps')

# --- Output: folder for globally normalized uncertainty maps ---
uncer_norm_dir = os.path.join(base_dir, COMBINED_FOLDER_NAME, 'uncertainty_maps_global_norm')
os.makedirs(uncer_norm_dir, exist_ok=True)

# --- Discover cases from existing raw uncertainty maps ---
# Assumes filenames like: _uncertainty.nii.gz
raw_files = sorted(glob(os.path.join(uncer_raw_dir, "*_uncertainty.nii.gz")))
case_ids = [
    os.path.basename(f).replace("_uncertainty.nii.gz", "")
    for f in raw_files
]
print(case_ids[:5])

print(f"Found {len(case_ids)} cases in:\n  {uncer_raw_dir}")

uncertainty_types = ["uncertainty"]  # Adjust if you have multiple types (e.g., aleatoric, epistemic, total)

# =============================================================================
# PASS 1: Scan all raw maps to compute global min/max and percentile stats
# =============================================================================
print("\n--- Pass 1: Computing global statistics from existing raw maps ---")

global_stats = {
    name: {"min": np.inf, "max": -np.inf}
    for name in uncertainty_types
}

percentile_tracker = {
    name: {"p1_vals": [], "p99_vals": [], "means": [], "stds": []}
    for name in uncertainty_types
}

# load global stats from the pkl file if available
# stats_path = os.path.join(uncer_norm_dir, "global_uncertainty_stats.pkl")
# if os.path.exists(stats_path):
#     with open(stats_path, 'rb') as f:
#         saved_stats = pickle.load(f)
#         global_stats = saved_stats.get("global_stats", global_stats)
#         percentile_tracker = saved_stats.get("percentile_tracker", percentile_tracker)
#     print(f"Loaded existing global stats from:\n  {stats_path}")

for case_id in case_ids:
    print(f"  [Pass 1] {case_id}")

    for name in uncertainty_types:
        raw_path = os.path.join(uncer_raw_dir, f"{case_id}_{name}.nii.gz")

        if not os.path.exists(raw_path):
            print(f"    WARNING: Missing {raw_path}, skipping.")
            continue

        raw_img = sitk.ReadImage(raw_path)
        array = sitk.GetArrayFromImage(raw_img).astype(np.float32)

        # Update global min/max
        arr_min, arr_max = float(np.min(array)), float(np.max(array))
        global_stats[name]["min"] = min(global_stats[name]["min"], arr_min)
        global_stats[name]["max"] = max(global_stats[name]["max"], arr_max)

        # Track per-case percentiles
        percentile_tracker[name]["p1_vals"].append(float(np.percentile(array, 1)))
        percentile_tracker[name]["p99_vals"].append(float(np.percentile(array, 99)))
        percentile_tracker[name]["means"].append(float(np.mean(array)))
        percentile_tracker[name]["stds"].append(float(np.std(array)))

# --- Compute normalization bounds ---
print("\n--- Global Statistics ---")
global_norm_bounds = {}

for name in uncertainty_types:
    gmin = global_stats[name]["min"]
    gmax = global_stats[name]["max"]

    # Robust bounds: median of per-case p1 / 95th percentile of per-case p99
    robust_low  = float(np.median(percentile_tracker[name]["p1_vals"]))
    robust_high = float(np.percentile(percentile_tracker[name]["p99_vals"], 95))

    print(f"  {name}:")
    print(f"    global_min   = {gmin:.6f},  global_max   = {gmax:.6f}")
    print(f"    robust_low   = {robust_low:.6f},  robust_high  = {robust_high:.6f}")

    # --- Choose normalization strategy (uncomment one) ---

    # Option A: strict global min/max
    global_norm_bounds[name] = (gmin, gmax)

    # Option B: robust percentile-based (clip outliers)
    # global_norm_bounds[name] = (robust_low, robust_high)

# Save stats for reproducibility
stats_path = os.path.join(uncer_norm_dir, "global_uncertainty_stats.pkl")
with open(stats_path, 'wb') as f:
    pickle.dump({
        "global_stats":        global_stats,
        "percentile_tracker":  percentile_tracker,
        "global_norm_bounds":  global_norm_bounds,
    }, f)
print(f"\n  Saved global stats to:\n    {stats_path}")

# =============================================================================
# PASS 2: Apply global normalization and save
# =============================================================================
print("\n--- Pass 2: Applying global normalization ---")

for case_id in case_ids:
    print(f"  [Pass 2] {case_id}")

    for name in uncertainty_types:
        raw_path = os.path.join(uncer_raw_dir, f"{case_id}_{name}.nii.gz")

        if not os.path.exists(raw_path):
            print(f"    WARNING: Missing {raw_path}, skipping.")
            continue

        # Load raw map
        raw_img = sitk.ReadImage(raw_path)
        array = sitk.GetArrayFromImage(raw_img).astype(np.float32)

        # Apply global normalization
        low, high = global_norm_bounds[name]
        if high > low:
            arr_norm = (array - low) / (high - low)
            arr_norm = np.clip(arr_norm, 0.0, 1.0)
        else:
            arr_norm = np.zeros_like(array)

        # Save normalized map (preserve original spatial metadata)
        norm_img = sitk.GetImageFromArray(arr_norm.astype(np.float32))
        norm_img.CopyInformation(raw_img)  # use raw_img directly — no need for separate ref

        out_path = os.path.join(uncer_norm_dir, f"{case_id}_{name}_uncertainty.nii.gz")
        sitk.WriteImage(norm_img, out_path)

print(f"\n✅ Done!")
print(f"   Normalized maps saved to:\n  {uncer_norm_dir}")
print(f"   Global stats saved to:\n  {stats_path}")