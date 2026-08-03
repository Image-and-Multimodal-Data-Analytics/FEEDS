import os
import glob

def find_matching_file(base_id, directory, pattern="*.nii.gz"):
    """Find files that contain the base_id in their filename"""
    search_pattern = os.path.join(directory, f"*{base_id}*{pattern}")
    matches = glob.glob(search_pattern)
    return matches[0] if matches else None

def find_matching_files(base_id, directory, extensions=None):
    """Find all files that contain the base_id with specified extensions"""
    if extensions is None:
        extensions = ["*.nii.gz", "*.npz"]
    
    matches = {}
    for ext in extensions:
        search_pattern = os.path.join(directory, f"*{base_id}*{ext}")
        files = glob.glob(search_pattern)
        matches[ext] = files[0] if files else None
    return matches