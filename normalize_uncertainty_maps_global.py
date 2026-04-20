"""
Normalize uncertainty maps globally across all files in a directory.

This script reads all uncertainty files from a given input directory,
computes global statistics (min/max and percentiles), and then applies
global normalization to produce normalized maps in an output directory.

Usage:
    python normalize_uncertainty_maps_global.py <input_dir> <output_dir> [--uncertainty-types aleatoric epistemic total]

Example:
    python normalize_uncertainty_maps_global.py "/path/to/uncertainty_maps" "/path/to/normalized_maps"
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
import pickle
from glob import glob


def discover_uncertainty_files(input_dir, uncertainty_types=None):
    """
    Discover all uncertainty files in the input directory.
    
    Searches for files matching: {case_id}_{uncertainty_type}_uncertainty.nii.gz
    
    Returns:
        case_ids: list of unique case identifiers
        uncertainty_types: list of uncertainty types found
    """
    if uncertainty_types is None:
        uncertainty_types = ["aleatoric", "epistemic", "total"]
    
    # Search for files matching pattern
    case_id_set = set()
    found_types = set()
    
    for unc_type in uncertainty_types:
        pattern = os.path.join(input_dir, f"*_{unc_type}_uncertainty.nii.gz")
        matching_files = glob(pattern)
        
        for f in matching_files:
            case_id = os.path.basename(f).replace(f"_{unc_type}_uncertainty.nii.gz", "")
            case_id_set.add(case_id)
            found_types.add(unc_type)
    
    case_ids = sorted(list(case_id_set))
    found_types = sorted(list(found_types))
    
    return case_ids, found_types


def compute_global_statistics(input_dir, case_ids, uncertainty_types):
    """
    PASS 1: Scan all uncertainty files to compute global statistics.
    
    Computes global min/max and percentile statistics for each uncertainty type.
    
    Returns:
        global_stats: dict with min/max for each uncertainty type
        percentile_tracker: dict with percentile and distribution statistics
    """
    print("\n--- PASS 1: Computing global statistics ---")
    
    global_stats = {
        name: {"min": np.inf, "max": -np.inf}
        for name in uncertainty_types
    }
    
    percentile_tracker = {
        name: {"p1_vals": [], "p99_vals": [], "means": [], "stds": []}
        for name in uncertainty_types
    }
    
    for i, case_id in enumerate(case_ids):
        print(f"  [{i+1}/{len(case_ids)}] {case_id}")
        
        for unc_type in uncertainty_types:
            file_path = os.path.join(input_dir, f"{case_id}_{unc_type}_uncertainty.nii.gz")
            
            if not os.path.exists(file_path):
                print(f"    WARNING: Missing {file_path}")
                continue
            
            # Load uncertainty map
            img = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(img).astype(np.float32)
            
            # Update global min/max
            arr_min = float(np.min(array))
            arr_max = float(np.max(array))
            global_stats[unc_type]["min"] = min(global_stats[unc_type]["min"], arr_min)
            global_stats[unc_type]["max"] = max(global_stats[unc_type]["max"], arr_max)
            
            # Track percentiles and distribution stats
            percentile_tracker[unc_type]["p1_vals"].append(float(np.percentile(array, 1)))
            percentile_tracker[unc_type]["p99_vals"].append(float(np.percentile(array, 99)))
            percentile_tracker[unc_type]["means"].append(float(np.mean(array)))
            percentile_tracker[unc_type]["stds"].append(float(np.std(array)))
    
    return global_stats, percentile_tracker


def compute_normalization_bounds(global_stats, percentile_tracker, uncertainty_types, method="global"):
    """
    Compute normalization bounds from global statistics.
    
    Args:
        method: "global" (strict min/max) or "robust" (percentile-based, outlier-resistant)
    
    Returns:
        global_norm_bounds: dict with (low, high) bounds for each uncertainty type
    """
    print("\n--- Global Statistics ---")
    
    global_norm_bounds = {}
    
    for name in uncertainty_types:
        gmin = global_stats[name]["min"]
        gmax = global_stats[name]["max"]
        
        # Robust bounds: median of per-case 1st percentile, 95th percentile of per-case 99th percentile
        robust_low = float(np.median(percentile_tracker[name]["p1_vals"]))
        robust_high = float(np.percentile(percentile_tracker[name]["p99_vals"], 95))
        
        print(f"  {name}:")
        print(f"    global_min={gmin:.6f}, global_max={gmax:.6f}")
        print(f"    robust_low={robust_low:.6f}, robust_high={robust_high:.6f}")
        
        if method == "robust":
            global_norm_bounds[name] = (robust_low, robust_high)
        else:  # global (default)
            global_norm_bounds[name] = (gmin, gmax)
    
    return global_norm_bounds


def apply_global_normalization(input_dir, output_dir, case_ids, uncertainty_types, global_norm_bounds):
    """
    PASS 2: Apply global normalization to all uncertainty files.
    
    Normalizes each uncertainty map using global bounds and saves to output directory.
    """
    print("\n--- PASS 2: Applying global normalization ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, case_id in enumerate(case_ids):
        print(f"  [{i+1}/{len(case_ids)}] {case_id}")
        
        for unc_type in uncertainty_types:
            input_path = os.path.join(input_dir, f"{case_id}_{unc_type}_uncertainty.nii.gz")
            output_path = os.path.join(output_dir, f"{case_id}_{unc_type}_uncertainty_normalized.nii.gz")
            
            if not os.path.exists(input_path):
                print(f"    WARNING: Missing {input_path}")
                continue
            
            # Load uncertainty map
            img = sitk.ReadImage(input_path)
            array = sitk.GetArrayFromImage(img).astype(np.float32)
            
            # Apply global normalization
            low, high = global_norm_bounds[unc_type]
            if high > low:
                arr_norm = (array - low) / (high - low)
                arr_norm = np.clip(arr_norm, 0.0, 1.0)
            else:
                arr_norm = np.zeros_like(array)
            
            # Save normalized map
            norm_img = sitk.GetImageFromArray(arr_norm.astype(np.float32))
            norm_img.CopyInformation(img)  # Preserve spatial metadata
            sitk.WriteImage(norm_img, output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Normalize uncertainty maps globally across all files in a directory."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing uncertainty files (e.g., case_id_aleatoric_uncertainty.nii.gz)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for globally normalized uncertainty files"
    )
    parser.add_argument(
        "--uncertainty-types",
        type=str,
        nargs="+",
        default=["aleatoric", "epistemic", "total"],
        help="Uncertainty types to normalize (default: aleatoric epistemic total)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["global", "robust"],
        default="global",
        help="Normalization method: 'global' (strict min/max) or 'robust' (percentile-based)"
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Save global statistics to a pickle file in the output directory"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    uncertainty_types = args.uncertainty_types
    method = args.method
    save_stats = args.save_stats
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Discover uncertainty files
    case_ids, found_types = discover_uncertainty_files(input_dir, uncertainty_types)
    
    if not case_ids:
        print(f"ERROR: No uncertainty files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(case_ids)} cases with {len(found_types)} uncertainty types:")
    print(f"  Case IDs: {case_ids[:5]}{'...' if len(case_ids) > 5 else ''}")
    print(f"  Uncertainty types: {found_types}")
    
    # Use only the types that were actually found
    uncertainty_types = found_types
    
    # PASS 1: Compute global statistics
    global_stats, percentile_tracker = compute_global_statistics(
        input_dir, case_ids, uncertainty_types
    )
    
    # Compute normalization bounds
    global_norm_bounds = compute_normalization_bounds(
        global_stats, percentile_tracker, uncertainty_types, method=method
    )
    
    # PASS 2: Apply normalization
    apply_global_normalization(
        input_dir, output_dir, case_ids, uncertainty_types, global_norm_bounds
    )
    
    # Save statistics if requested
    if save_stats:
        stats_path = os.path.join(output_dir, "global_uncertainty_stats.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump({
                "global_stats": global_stats,
                "percentile_tracker": percentile_tracker,
                "global_norm_bounds": global_norm_bounds,
                "uncertainty_types": uncertainty_types,
                "method": method
            }, f)
        print(f"\n  Saved statistics to:\n    {stats_path}")
    
    print(f"\n✅ Done!")
    print(f"   Input directory:  {input_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Normalization method: {method}")
    print(f"   Processed {len(case_ids)} cases with {len(uncertainty_types)} uncertainty types")


if __name__ == "__main__":
    main()
