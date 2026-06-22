# Bone Analysis

Tooling for evaluating PET/CT lesion segmentation predictions against ground truth, with a focus on lesion-level and bone-metastasis analysis. Includes both a lightweight per-file FP/FN/Dice metrics generator and a fuller lesion/mets analysis pipeline.

## Contents

| File | Purpose |
|------|---------|
| `main.py` | Unified entry point for lesion and bone-metastasis analysis. Replaces the older `probability_analysis.py` and `main_mets.py`. |
| `Get_fp_fn_dice.py` | Standalone script that computes per-case Dice, false-positive volume, false-negative volume, and lesion counts using connected-component analysis. |
| `process.py` | Worker module providing `process_image` and `process_bone_mets` (consumed by `main.py`). |
| `lesion_metrics.sh` | Runs the file-wise Dice / FP / FN metrics generation. |
| `lesion_metrics_array.sh` | Generates a `*_mets.csv` and `*_mets_metrics.csv` file for each case in the input directory (array job). |
| `Get_fp_fn_dice.sh` | Wrapper that runs the FP / FN / Dice CSV generation. |

> Note: `process.py`, `ts_table.csv` (organ map), and the exact `.py` filenames invoked by each `.sh` script should be confirmed against the repo — they are inferred here from the two main scripts.

## Requirements

- Python 3.8+
- `numpy`, `nibabel`, `cc3d` (connected-components-3d), `tqdm`
- `pandas` (used by the mets pipeline for CSV export)

```bash
pip install numpy nibabel connected-components-3d tqdm pandas
```

All inputs are expected as `.nii.gz` volumes, with prediction and ground-truth files sharing the same filename across their respective directories.

## Usage

The main operations are organized into bash scripts; input variables (paths, fold, mode) can be edited at the top of each script.

### 1. Per-case FP / FN / Dice metrics

Computes Dice, false-positive volume (mL), false-negative volume (mL), and missed/detected lesion counts per case using 18-connectivity connected components, writing one CSV row per file.

```bash
bash Get_fp_fn_dice.sh
```

Or directly:

```bash
python Get_fp_fn_dice.py \
    --pred_dir /path/to/predictions \
    --gt_dir   /path/to/ground_truth \
    --save_dir /path/to/output.csv \
    --num_workers 8
```

Output CSV columns: `filename, dice_score, false_pos_vol, false_neg_vol, false_missed_lesions, false_detected_lesions, num_gt_lesions, num_pred_lesions`. A `dice_score` of `None` indicates no overlap between prediction and ground truth.

### 2. Lesion / bone-metastasis analysis

The unified `main.py` supports three modes: `lesion`, `mets`, or `both`.

```bash
# Lesion analysis
python main.py --mode lesion -f comb \
    -i /path/images -ld /path/labels -b /path/bone -p /path/preds

# Bone-metastasis analysis (one CSV pair per case)
python main.py --mode mets -f fold_0 \
    -i /path/images -ld /path/labels -b /path/bone -p /path/preds

# Both
python main.py --mode both \
    -i /path/images -ld /path/labels -b /path/bone -p /path/preds
```

For batch/array submission across all cases:

```bash
bash lesion_metrics_array.sh
```

#### `main.py` arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `both` | `lesion`, `mets`, or `both`. |
| `-k`, `--key` | `*` | Glob key to filter input cases. |
| `-f`, `--fold` | `comb` | Fold identifier. |
| `-j`, `--workers` | `4` | Parallel worker processes. |
| `-s`, `--skip-existing` | off | Skip cases already processed (resumable). |
| `-i`, `--image-dir` | required | PET/CT image volumes. |
| `-ld`, `--label-dir` | required | Ground-truth label volumes. |
| `-b`, `--bone-dir` | required | Bone masks. |
| `-p`, `--pred-dir` | required | Prediction volumes. |
| `-u`, `--uncertainty-dir` | `None` | Optional uncertainty maps. |
| `-o`, `--save-dir` | `./results` | Output root (results written under `bone_analysis/`). |
| `--organ_map` | `./ts_table.csv` | Organ lookup table. |

## Outputs

- **FP/FN/Dice script:** a single CSV with one row per case.
- **Lesion mode:** a checkpointed pickle (`summary_<key>.pkl`) holding per-case prediction, ground-truth, and image statistics. Checkpoints are written every 10 files and the run is resumable via `--skip-existing`.
- **Mets mode:** per-case `<image_id>_mets.csv` and `<image_id>_mets_metrics.csv` files.

## Reference paths

- Cluster results: `/dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/nnUNet/bone_analysis`
- Repo: `Image-and-Multimodal-Data-Analytics/semi-supervised-learning` → `bone_analysis/`
