# collate_results.py
"""
Collates per-image results from main.py outputs into summary CSV tables.

Output CSVs (all written to --out-dir):
  summary_overall.csv          – per-fold image-level dice / FP / FN
  summary_by_tracer.csv        – same split by FDG / PSMA
  summary_lesion_pred.csv      – predicted-lesion level stats
  summary_lesion_gt.csv        – GT-lesion level stats
  summary_organ.csv            – per-organ overlap (all folds)
  summary_bone.csv             – per-bone overlap  (all folds)
  summary_high_risk.csv        – high-risk segment sensitivity (all folds)
  summary_liver.csv            – liver-specific stats (all folds)

Usage
-----
python collate_results.py \\
    --results-dir /path/to/results \\   # folder that contains fold_0/ fold_1/ …
    --folds fold_0 fold_1 fold_2 \\
    --out-dir ./collated
"""

import os
import glob
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── TotalSegmentator label IDs ────────────────────────────────────────────────
LIVER_LABEL      = 5          # adjust if your ts_table differs
ORGAN_LABELS     = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,21,22,23,24,90,51]
BONE_LABELS      = [25,26,27,31,32,43,44,69,70,71,72,73,74,75,76,77,78]

# Label → human-readable name (subset; extend as needed)
LABEL_NAMES = {
    1:"spleen", 2:"kidney_right", 3:"kidney_left", 4:"gallbladder",
    5:"liver", 6:"stomach", 7:"pancreas", 8:"adrenal_gland_right",
    9:"adrenal_gland_left", 10:"lung_upper_lobe_left",
    11:"lung_lower_lobe_left", 12:"lung_upper_lobe_right",
    13:"lung_middle_lobe_right", 14:"lung_lower_lobe_right",
    21:"esophagus", 22:"trachea", 23:"thyroid_gland",
    24:"small_bowel", 90:"colon", 51:"urinary_bladder",
    25:"vertebrae_L5", 26:"vertebrae_L4", 27:"vertebrae_L3",
    31:"vertebrae_T12", 32:"vertebrae_T11",
    43:"rib_left_1", 44:"rib_left_2",
    69:"hip_left", 70:"hip_right",
    71:"femur_left", 72:"femur_right",
    73:"humerus_left", 74:"humerus_right",
    75:"scapula_left", 76:"scapula_right",
    77:"clavicula_left", 78:"clavicula_right",
}


# ═════════════════════════════════════════════════════════════════════════════
# Loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_pickle_summary(pkl_path: str) -> list[dict]:
    """Load a summary pickle produced by main.py (list of per-image dicts)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):          # older format: dict keyed by id
        data = list(data.values())
    return data or []


def load_mets_csvs(fold_dir: str) -> pd.DataFrame:
    """
    Concatenate all *_mets.csv files in fold_dir into one DataFrame.
    Adds an 'image_id' column derived from the filename.
    """
    parts = []
    for path in glob.glob(os.path.join(fold_dir, "*_mets.csv")):
        df = pd.read_csv(path)
        df.insert(0, "image_id",
                  os.path.basename(path).replace("_mets.csv", ""))
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_mets_metrics_csvs(fold_dir: str) -> pd.DataFrame:
    """Concatenate all *_mets_metrics.csv files."""
    parts = []
    for path in glob.glob(os.path.join(fold_dir, "*_mets_metrics.csv")):
        df = pd.read_csv(path)
        df.insert(0, "image_id",
                  os.path.basename(path).replace("_mets_metrics.csv", ""))
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def tracer(image_id: str) -> str:
    lid = str(image_id).lower()
    if "fdg"  in lid: return "FDG"
    if "psma" in lid: return "PSMA"
    return "Unknown"


def _mean_ci(series: pd.Series) -> dict:
    """Return mean ± 95 % CI for a numeric series (NaN-safe)."""
    s   = series.dropna()
    n   = len(s)
    mu  = s.mean() if n else np.nan
    sem = s.sem()  if n > 1 else np.nan
    return {"mean": mu, "std": s.std() if n > 1 else np.nan,
            "ci95_lo": mu - 1.96*sem if n > 1 else np.nan,
            "ci95_hi": mu + 1.96*sem if n > 1 else np.nan,
            "n": n}


def _agg(df: pd.DataFrame, col: str) -> dict:
    return _mean_ci(df[col]) if col in df.columns else {}


# ═════════════════════════════════════════════════════════════════════════════
# (A) Image-level summary from pickle
# ═════════════════════════════════════════════════════════════════════════════

def build_image_level_df(records: list[dict], fold: str) -> pd.DataFrame:
    """
    Flatten image_stats dicts from the pickle into one row per image.
    """
    rows = []
    for rec in records:
        stats = rec.get("image_stats") or {}
        if not stats:
            continue
        row = {"fold": fold, "image_id": rec.get("id", stats.get("image_id", ""))}
        row.update(stats)
        row["tracer"] = tracer(row["image_id"])
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# (B) Lesion-level summary from pickle
# ═════════════════════════════════════════════════════════════════════════════

def build_lesion_level_df(records: list[dict], fold: str,
                          kind: str = "pred") -> pd.DataFrame:
    """kind ∈ {'pred', 'gt'}"""
    key = "pred_stats" if kind == "pred" else "ground_stats"
    parts = []
    for rec in records:
        df = rec.get(key)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            df = pd.DataFrame([df])
        df.insert(0, "fold", fold)
        df["tracer"] = df["image_id"].apply(tracer) if "image_id" in df.columns else "Unknown"
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# (C) Organ / bone / high-risk / liver from mets CSVs
# ═════════════════════════════════════════════════════════════════════════════

def build_segment_df(mets_df: pd.DataFrame, fold: str,
                     label_ids: list[int], seg_type: str) -> pd.DataFrame:
    """
    Filter mets_df to rows matching label_ids, add fold / tracer / label name.
    seg_type ∈ {'organ', 'bone'}
    """
    if mets_df.empty or "Label" not in mets_df.columns:
        return pd.DataFrame()
    df = mets_df[mets_df["Label"].isin(label_ids)].copy()
    df.insert(0, "fold", fold)
    df["segment_type"] = seg_type
    df["label_name"]   = df["Label"].map(LABEL_NAMES).fillna(df["Label"].astype(str))
    df["tracer"]       = df["image_id"].apply(tracer)
    return df


def build_high_risk_df(mets_df: pd.DataFrame,
                       metrics_df: pd.DataFrame,
                       fold: str) -> pd.DataFrame:
    """
    High-risk analysis:
      – per-image sensitivity / dice from metrics CSV
      – counts of HR segments (ground vs pred) from mets CSV
    """
    rows = []

    # from metrics CSV
    if not metrics_df.empty:
        for _, r in metrics_df.iterrows():
            rows.append({
                "fold":               fold,
                "image_id":           r.get("image_id", ""),
                "tracer":             tracer(str(r.get("image_id", ""))),
                "dice":               r.get("dice",               np.nan),
                "sensitivity_all":    r.get("sensitivity",        np.nan),
                "sensitivity_organs": r.get("sensitivity organs", np.nan),
                "sensitivity_bones":  r.get("sensitivity bones",  np.nan),
            })

    if not rows:
        return pd.DataFrame()

    hr_df = pd.DataFrame(rows)

    # optionally enrich with HR counts from mets_df
    if not mets_df.empty and "High Risk_ground" in mets_df.columns:
        agg = (mets_df
               .groupby("image_id")
               .agg(
                   HR_ground_count=("High Risk_ground", "sum"),
                   HR_pred_count=("High Risk_pred",   "sum"),
               )
               .reset_index())
        hr_df = hr_df.merge(agg, on="image_id", how="left")

    return hr_df


def build_liver_df(mets_df: pd.DataFrame, fold: str) -> pd.DataFrame:
    """Extract liver-specific rows (label == LIVER_LABEL)."""
    if mets_df.empty or "Label" not in mets_df.columns:
        return pd.DataFrame()
    df = mets_df[mets_df["Label"] == LIVER_LABEL].copy()
    df.insert(0, "fold", fold)
    df["tracer"] = df["image_id"].apply(tracer)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# (D) Aggregate helper → one-row-per-fold summary
# ═════════════════════════════════════════════════════════════════════════════

_IMAGE_COLS = [
    "Dice_coefficient", "Sensitivity_voxel", "PPV_voxel",
    "FN_volume_mm3", "FP_volume_mm3", "TP_volume_mm3",
    "GT_lesions", "Predicted_lesions",
]

def summarise_image_level(img_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one aggregated row per (fold, tracer) combination,
    plus an 'All' tracer row.
    """
    if img_df.empty:
        return pd.DataFrame()

    def _agg_group(df, fold, tracer_label):
        row = {"fold": fold, "tracer": tracer_label, "n": len(df)}
        for col in _IMAGE_COLS:
            if col in df.columns:
                stats = _mean_ci(df[col])
                for k, v in stats.items():
                    row[f"{col}_{k}"] = v
        return row

    rows = []
    for fold, gf in img_df.groupby("fold"):
        rows.append(_agg_group(gf, fold, "All"))
        for tr, gt in gf.groupby("tracer"):
            rows.append(_agg_group(gt, fold, tr))

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Collate main.py results into CSVs")
    p.add_argument("--results-dir", required=True,
                   help="Root dir containing fold_X/ subdirectories")
    p.add_argument("--folds", nargs="+",
                   default=["fold_0", "fold_1", "fold_2", "comb"],
                   help="Fold sub-directory names to process")
    p.add_argument("--out-dir", default="./collated",
                   help="Where to write output CSVs")
    p.add_argument("--no-lesion-pickle", action="store_true",
                   help="Skip loading pickle (only use mets CSVs)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # accumulate across folds
    all_image       = []
    all_pred_lesion = []
    all_gt_lesion   = []
    all_organ       = []
    all_bone        = []
    all_high_risk   = []
    all_liver       = []

    for fold in args.folds:
        fold_dir = os.path.join(args.results_dir, fold)
        if not os.path.isdir(fold_dir):
            print(f"  [skip] {fold_dir} not found")
            continue

        print(f"\n── {fold} ──────────────────────────────")

        # ── (1) pickle (lesion analysis) ─────────────────────────────
        if not args.no_lesion_pickle:
            pkl_candidates = glob.glob(os.path.join(fold_dir, "summary_*.pkl"))
            if not pkl_candidates:
                # try one level up
                pkl_candidates = glob.glob(
                    os.path.join(args.results_dir, f"summary_*{fold}*.pkl"))

            if pkl_candidates:
                pkl_path = pkl_candidates[0]
                print(f"  Loading pickle: {pkl_path}")
                records = load_pickle_summary(pkl_path)
                print(f"  {len(records)} records found")

                img_df  = build_image_level_df(records, fold)
                pred_df = build_lesion_level_df(records, fold, "pred")
                gt_df   = build_lesion_level_df(records, fold, "gt")

                all_image.append(img_df)
                all_pred_lesion.append(pred_df)
                all_gt_lesion.append(gt_df)
            else:
                print(f"  No pickle found in {fold_dir}")

        # ── (2) mets CSVs ────────────────────────────────────────────
        mets_df    = load_mets_csvs(fold_dir)
        metrics_df = load_mets_metrics_csvs(fold_dir)
        print(f"  Mets rows: {len(mets_df)}  |  Metrics rows: {len(metrics_df)}")

        if not mets_df.empty:
            all_organ.append(
                build_segment_df(mets_df, fold, ORGAN_LABELS, "organ"))
            all_bone.append(
                build_segment_df(mets_df, fold, BONE_LABELS, "bone"))
            all_liver.append(build_liver_df(mets_df, fold))

        all_high_risk.append(
            build_high_risk_df(mets_df, metrics_df, fold))

    # ── write CSVs ────────────────────────────────────────────────────

    def _save(frames: list, name: str, summarise_fn=None):
        if not frames:
            print(f"  [skip] no data for {name}")
            return
        combined = pd.concat(
            [f for f in frames if f is not None and not f.empty],
            ignore_index=True)
        if combined.empty:
            print(f"  [skip] empty frame for {name}")
            return
        path = os.path.join(args.out_dir, name)
        combined.to_csv(path, index=False)
        print(f"  → {path}  ({len(combined)} rows)")

        if summarise_fn is not None:
            summ = summarise_fn(combined)
            if summ is not None and not summ.empty:
                spath = path.replace(".csv", "_aggregated.csv")
                summ.to_csv(spath, index=False)
                print(f"  → {spath}  ({len(summ)} rows)")

    print("\n── Writing CSVs ─────────────────────────────────────────────")
    _save(all_image,       "summary_overall.csv",       summarise_image_level)
    _save(all_pred_lesion, "summary_lesion_pred.csv",   _summarise_detection)
    _save(all_gt_lesion,   "summary_lesion_gt.csv",     _summarise_detection)
    _save(all_organ,       "summary_organ.csv",         _summarise_segment)
    _save(all_bone,        "summary_bone.csv",          _summarise_segment)
    _save(all_high_risk,   "summary_high_risk.csv",     _summarise_high_risk)
    _save(all_liver,       "summary_liver.csv",         _summarise_liver)

    # ── bonus: pretty printed console table ──────────────────────────
    _print_console_table(all_image)


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation helpers for _save()
# ═════════════════════════════════════════════════════════════════════════════

def _summarise_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lesion-level TP/FP/FN counts and average probs/uncertainties."""
    if df.empty:
        return pd.DataFrame()

    det_col = "Detection" if "Detection" in df.columns else None
    rows = []
    for (fold, tracer_lbl), g in df.groupby(["fold", "tracer"]):
        row = {"fold": fold, "tracer": tracer_lbl, "n_images": g["image_id"].nunique(),
               "n_lesions": len(g)}
        if det_col:
            row["TP"] = int((g[det_col] == "TP").sum())
            row["FP"] = int((g[det_col] == "FP").sum())
            row["FN"] = int((g[det_col] == "FN").sum())
            denom = row["TP"] + row["FN"]
            row["lesion_sensitivity"] = row["TP"] / denom if denom else np.nan
            denom2 = row["TP"] + row["FP"]
            row["lesion_PPV"] = row["TP"] / denom2 if denom2 else np.nan

        for col in ["gt_lesion_volume", "pred_lesion_volume",
                    "avg_prob_lesion_gt", "avg_total_uncertainty_gt",
                    "avg_epistemic_uncertainty_gt", "avg_aleatoric_uncertainty_gt",
                    "avg_prob_pred_lesion", "avg_total_uncertainty_pred"]:
            if col in g.columns:
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"]  = g[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Per-label mean overlap across all images, split by fold and tracer."""
    if df.empty or "label_name" not in df.columns:
        return pd.DataFrame()

    agg_cols = [c for c in
                ["Overlap_Percentage_ground", "Overlap_Percentage_pred",
                 "Overlap_Volume_ground",     "Overlap_Volume_pred",
                 "Overlap_Count_ground",      "Overlap_Count_pred"]
                if c in df.columns]
    if not agg_cols:
        # try without suffix
        agg_cols = [c for c in
                    ["Overlap_Percentage", "Overlap_Volume", "Overlap_Count"]
                    if c in df.columns]

    if not agg_cols:
        return pd.DataFrame()

    rows = []
    for (fold, tracer_lbl, lname), g in df.groupby(
            ["fold", "tracer", "label_name"]):
        row = {"fold": fold, "tracer": tracer_lbl,
               "label_name": lname, "n": len(g)}
        for col in agg_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"]  = g[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-risk sensitivity metrics across folds / tracers."""
    if df.empty:
        return pd.DataFrame()

    metric_cols = [c for c in
                   ["dice", "sensitivity_all", "sensitivity_organs",
                    "sensitivity_bones", "HR_ground_count", "HR_pred_count"]
                   if c in df.columns]
    rows = []
    for (fold, tracer_lbl), g in df.groupby(["fold", "tracer"]):
        row = {"fold": fold, "tracer": tracer_lbl, "n": len(g)}
        for col in metric_cols:
            stats = _mean_ci(g[col])
            for k, v in stats.items():
                row[f"{col}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise_liver(df: pd.DataFrame) -> pd.DataFrame:
    """Liver-specific aggregation."""
    if df.empty:
        return pd.DataFrame()

    ovlp_cols = [c for c in df.columns if "Overlap" in c or "High Risk" in c]
    rows = []
    for (fold, tracer_lbl), g in df.groupby(["fold", "tracer"]):
        row = {"fold": fold, "tracer": tracer_lbl, "n": len(g)}
        for col in ovlp_cols:
            if pd.api.types.is_numeric_dtype(g[col]):
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"]  = g[col].std()
        if "High Risk_ground" in g.columns:
            row["liver_HR_ground_pct"] = g["High Risk_ground"].mean() * 100
        if "High Risk_pred" in g.columns:
            row["liver_HR_pred_pct"]   = g["High Risk_pred"].mean() * 100
            if "High Risk_ground" in g.columns:
                tp = (g["High Risk_ground"] & g["High Risk_pred"]).sum()
                fn = (g["High Risk_ground"] & ~g["High Risk_pred"]).sum()
                row["liver_sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Console pretty-print
# ═════════════════════════════════════════════════════════════════════════════

def _print_console_table(all_image_frames: list):
    if not all_image_frames:
        return
    img_df = pd.concat(
        [f for f in all_image_frames if f is not None and not f.empty],
        ignore_index=True)
    if img_df.empty:
        return

    summ = summarise_image_level(img_df)
    if summ.empty:
        return

    dice_col = "Dice_coefficient_mean"
    fp_col   = "FP_volume_mm3_mean"
    fn_col   = "FN_volume_mm3_mean"

    display_cols = ["fold", "tracer", "n",
                    dice_col, fp_col, fn_col]
    display_cols = [c for c in display_cols if c in summ.columns]

    out = summ[display_cols].copy()
    if dice_col in out.columns:
        out[dice_col] = (out[dice_col] * 100).round(2)   # → percentage
    for c in [fp_col, fn_col]:
        if c in out.columns:
            out[c] = out[c].round(3)

    rename = {
        dice_col: "DICE (%)",
        fp_col:   "FP vol (mm³)",
        fn_col:   "FN vol (mm³)",
    }
    out.rename(columns=rename, inplace=True)

    SEP = "=" * 90
    print(f"\n{SEP}")
    print("SUMMARY TABLE  (Dice = mean over images; FP/FN = mean volume per image)")
    print(SEP)
    print(out.to_string(index=False))
    print(SEP)


if __name__ == "__main__":
    main()