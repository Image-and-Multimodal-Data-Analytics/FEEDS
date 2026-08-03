import os
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def convert_file(nii_path: Path, output_dir: Path) -> str:
    try:
        img  = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)  # shape: (X, Y, Z) from nibabel

        # nibabel loads as (X, Y, Z) → reorder to (Z, X, Y)
        data = np.transpose(data, (2, 0, 1))    # (X, Y, Z) → (Z, X, Y)

        # Add channel dim → (C, Z, X, Y)
        data = data[np.newaxis, ...]             # (Z, X, Y) → (1, Z, X, Y)

        print(f"   shape: {data.shape} | dtype: {data.dtype}")

        # Build output path
        stem     = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}_seg_org.npy"

        np.save(str(out_path), data)
        return f"✅ {nii_path.name} → {out_path.name}  shape={data.shape}"

    except Exception as e:
        return f"❌ {nii_path.name} FAILED: {e}"


def batch_convert(input_dir: str, output_dir: str, workers: int = 8):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all .nii and .nii.gz files
    files = list(input_path.rglob("*.nii.gz")) + list(input_path.rglob("*.nii"))
    files = list(set(files))  # deduplicate

    print(f"Found {len(files)} files. Converting with {workers} workers...\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_file, f, output_path): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert .nii/.nii.gz to .npy (C,Z,X,Y)")
    parser.add_argument("--input",   "-i", required=False, help="Input directory with .nii files")
    parser.add_argument("--output",  "-o", required=False, help="Output directory for .npy files")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Number of parallel workers")
    args = parser.parse_args()

    args.input  = '../../nnUNet_data/nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_10/train_predictions/'
    args.output = args.input

    batch_convert(args.input, args.output, args.workers)