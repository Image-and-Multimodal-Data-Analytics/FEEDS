"""
feeds_stats_core.py
===================================================================
Core statistics for the FEEDS manuscript.

Produces the manuscript comparison table:
  * AutoPET  : per-diagnosis rows (Dice/FNVol on diseased diagnoses;
               FPVol also on Negative)
  * DeepPSMA : pooled row
  * DH       : pooled row

Two families of tests:
  (1) SUPERIORITY  FEEDS vs random (paired Wilcoxon + t).
      FDR-corrected PER METRIC, POOLED ACROSS ALL DATASETS
      (Benjamini-Hochberg, via statsmodels). Correction runs on the
      Wilcoxon p-value.
  (2) NON-INFERIORITY  FEEDS vs 100% labeled.  SEPARATE family,
      NOT included in the correction (reported as a pass/fail flag).

Random iterations are summarized per case as BOTH mean and median.
All data access mirrors stat_analysis.ipynb / evaluation_with_statistics.

NOTE ON CORRECTION: Holm/BH need the WHOLE set of p-values at once, so
the table is assembled first, then corrected per metric. The number of
tests per family is simply the number of assembled rows (printed).
===================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# Multiplicity correction uses statsmodels (see correct_family()).
try:
    from statsmodels.stats.multitest import multipletests
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False


TRAINER = "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres"


# ------------------------------------------------------------------ paths
def _root():
    return os.environ["nnUNet_results"]


def summary_path(dataset_num, fold, split):
    return os.path.join(
        _root(), f"Dataset{dataset_num}_AutoPet", TRAINER, fold, split, "summary.json"
    )


# ------------------------------------------------------------------ metadata
def convert_location_to_case_name(file_locations):
    names = []
    for path in file_locations:
        parts = path.split("/")
        patient_id = parts[2].replace("PETCT_", "")
        scan_folder = parts[3]
        names.append(f"fdg_{patient_id}_{scan_folder}")
    return names


def _diagnosis_lookup(metadata_csv="fdg_metadata.csv"):
    if not os.path.exists(metadata_csv):
        print(f"[warn] {metadata_csv} not found -> FDG diagnosis will be NaN")
        return {}
    md = pd.read_csv(metadata_csv)
    md["file"] = convert_location_to_case_name(md["File Location"].values)
    md = md[["file", "diagnosis"]].drop_duplicates(subset=["file"])
    return dict(zip(md["file"], md["diagnosis"]))


def _tracer(case_name):
    cl = case_name.lower()
    return "FDG" if cl.startswith("fdg") else "PSMA" if cl.startswith("psma") else None


# ------------------------------------------------------------------ loader
def load_arm(dataset_num, folds, split, metadata_csv="fdg_metadata.csv",
             dataset_tag=None):
    """Per-case voxel-level dataframe for (dataset, folds, split).

    One row per (fold, case): dataset_tag, fold, case_name, tracer,
    diagnosis, diseased, Dice, false_pos_vol, false_neg_vol.
    FNVol is NaN for non-diseased cases (undefined without a lesion).
    """
    dx = _diagnosis_lookup(metadata_csv)
    rows = []
    for fold in folds:
        sp = summary_path(dataset_num, fold, split)
        if not os.path.exists(sp):
            print(f"[skip] missing {sp}")
            continue
        summary = json.load(open(sp))
        lpath = sp.replace("summary.json", "lesion_metrics.csv")
        lez = None
        if os.path.exists(lpath):
            lez = pd.read_csv(lpath)
            lez["filename"] = lez["filename"].astype(str).str.replace(
                ".nii.gz", "", regex=False)
            lez = lez.set_index("filename")
        for case in summary["metric_per_case"]:
            m = case["metrics"]["1"]
            name = os.path.basename(case["prediction_file"]).replace(".nii.gz", "")
            diseased = (m["TP"] + m["FN"]) > 0
            diag = ("PROSTATE_CANCER" if name.lower().startswith("psma")
                    else dx.get(name))
            fp_vol = fn_vol = np.nan
            if lez is not None and name in lez.index:
                lr = lez.loc[name]
                fp_vol = lr.get("false_pos_vol", np.nan)
                fn_vol = lr.get("false_neg_vol", np.nan)
            rows.append(dict(
                dataset_tag=dataset_tag or f"D{dataset_num}",
                fold=fold, case_name=name, tracer=_tracer(name),
                diagnosis=diag, diseased=bool(diseased),
                Dice=m.get("Dice", np.nan), false_pos_vol=fp_vol,
                false_neg_vol=(np.nan if not diseased else fn_vol),
            ))
    df = pd.DataFrame(rows)
    if not df.empty:
        neg = df["diagnosis"].eq("NEGATIVE")
        df.loc[neg, "diseased"] = False
        df.loc[neg, "false_neg_vol"] = np.nan
    return df


# ------------------------------------------------------------------ pooling (mean + median)
def pool_random_by_case(random_df):
    """Collapse the 5 random iterations to ONE row per case, keeping the
    per-case MEAN (base column) and per-case MEDIAN (col + '__median')
    of each metric, plus the per-case SD of the mean (col + '__sd') for
    the 'Random' mean +/- sd display in the table."""
    metric_cols = ["Dice", "false_pos_vol", "false_neg_vol"]
    meta = [c for c in ["dataset_tag", "tracer", "diagnosis", "diseased"]
            if c in random_df.columns]
    g = random_df.groupby("case_name", dropna=False)
    out = g.agg(**{c: (c, "first") for c in meta}).reset_index()
    for m in metric_cols:
        out[m] = g[m].mean().values
        out[m + "__median"] = g[m].median().values
        out[m + "__sd"] = g[m].std(ddof=1).values
    out["fold"] = "random_summary"
    return out


# ------------------------------------------------------------------ metric spec
METRICS = [
    # (column,          pretty,     higher_is_better,  diseased_only)
    ("Dice",          "Dice",     True,   True),
    ("false_pos_vol", "FP Vol",   False,  False),
    ("false_neg_vol", "FN Vol",   False,  True),
]


def _colname(col, summary):
    return col if summary == "mean" else col + "__median"


def _pair(f_df, c_df, f_col, c_col, diseased_only):
    """Paired (feeds, comparator) arrays matched on case_name, NaNs dropped."""
    fcols = ["case_name", f_col] + (["diseased"] if "diseased" in f_df else [])
    ccols = ["case_name", c_col] + (["diseased"] if "diseased" in c_df else [])
    fa = f_df[fcols].rename(columns={f_col: "_f"})
    cb = c_df[ccols].rename(columns={c_col: "_c"})
    m = fa.merge(cb, on="case_name", suffixes=("_ff", "_cc"))
    if diseased_only:
        for dc in ["diseased_ff", "diseased_cc", "diseased"]:
            if dc in m:
                m = m[m[dc]]
    m = m.dropna(subset=["_f", "_c"])
    return m["_f"].to_numpy(float), m["_c"].to_numpy(float)


# ------------------------------------------------------------------ single paired tests
def paired_superiority(feeds_df, comp_df, col, hib, dis_only, summary="mean"):
    """One paired superiority test -> dict of stats (or None if n<2)."""
    c_col = _colname(col, summary)
    if c_col not in comp_df.columns:
        c_col = col
    a, b = _pair(feeds_df, comp_df, col, c_col, dis_only)
    n = len(a)
    if n < 2:
        return None
    diff = a - b
    t, p_t = stats.ttest_rel(a, b)
    try:
        _, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = np.nan
    # comparator sd for the "Random mean +/- sd" display
    sd_col = col + "__sd"
    comp_sd = comp_df[sd_col].mean() if sd_col in comp_df.columns else np.nan
    return dict(
        n=n, feeds=a.mean(), comp=b.mean(), comp_sd=comp_sd,
        delta=diff.mean(), t=t, p_ttest=p_t, p_wilcoxon=p_w,
        favored=("FEEDS" if ((diff.mean() < 0) != hib) else "comparator"),
    )


def paired_noninferiority(feeds_df, ref_df, col, hib, dis_only, margin,
                          summary="mean", alpha=0.05):
    """One paired non-inferiority test -> dict (or None if n<2).

    HIGHER better: non-inferior if mean(FEEDS-ref) > -margin
        t=(md+margin)/se, p=P(T>=t).
    LOWER better: non-inferior if mean(FEEDS-ref) < margin
        t=(md-margin)/se, p=P(T<=t).
    """
    r_col = _colname(col, summary)
    if r_col not in ref_df.columns:
        r_col = col
    a, b = _pair(feeds_df, ref_df, col, r_col, dis_only)
    n = len(a)
    if n < 2:
        return None
    diff = a - b
    md = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    dfree = n - 1
    if se == 0:
        t_stat = np.inf
        p_ni = 0.0 if ((hib and md > -margin) or (not hib and md < margin)) else 1.0
    elif hib:
        t_stat = (md + margin) / se
        p_ni = stats.t.sf(t_stat, dfree)
    else:
        t_stat = (md - margin) / se
        p_ni = stats.t.cdf(t_stat, dfree)
    return dict(n=n, delta=md, margin=margin, t_ni=t_stat,
                p_noninf=p_ni, noninferior=bool(p_ni < alpha))


# ------------------------------------------------------------------ multiplicity correction
def correct_family(pvals, method="fdr_bh", alpha=0.05):
    """Correct a family of p-values. Uses statsmodels.multipletests.

    Returns (corrected_pvals, reject_flags). NaNs are held out and
    returned as NaN/False (they are not counted in the family size).
    """
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    corr = np.full_like(p, np.nan)
    rej = np.zeros(len(p), dtype=bool)
    if ok.sum() == 0:
        return corr, rej
    if not _HAVE_SM:
        raise ImportError(
            "statsmodels is required for correction. "
            "pip install statsmodels  (method='fdr_bh' or 'holm')."
        )
    reject, p_adj, _, _ = multipletests(p[ok], alpha=alpha, method=method)
    corr[ok] = p_adj
    rej[ok] = reject
    return corr, rej


# ------------------------------------------------------------------ table builder
# DATASET PLAN: how each dataset contributes rows.
#   layout 'diagnosis' -> one row per diagnosis (paper's AutoPET block)
#   layout 'pooled'    -> a single pooled row (DeepPSMA, DH)
PLAN_SPLITS = {
    "AutoPET":  "test_predictions",
    "DeepPSMA": "test_dpsma_predictions",
    "DH":       "DHMC_test_predictions",
}
PLAN_LAYOUT = {
    "AutoPET":  "diagnosis",
    "DeepPSMA": "pooled",
    "DH":       "pooled",
}
# arm -> (dataset_number, folds, aggregation)
PLAN_ARMS = dict(
    feeds =dict(dataset=111, folds=["fold_4"], agg="single"),
    random=dict(dataset=330, folds=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
                agg="iterations"),
    full  =dict(dataset=111, folds=["fold_9"], agg="single"),
)

# nice display order/labels for diagnoses
DIAG_ORDER = ["LUNG_CANCER", "LYMPHOMA", "MELANOMA", "NEGATIVE", "PROSTATE_CANCER"]
DIAG_LABEL = {
    "LUNG_CANCER": "Lung Cancer", "LYMPHOMA": "Lymphoma",
    "MELANOMA": "Melanoma", "NEGATIVE": "Negative",
    "PROSTATE_CANCER": "Prostate Cancer",
}


def _load_plan_arm(tag, arm, split, metadata_csv):
    spec = PLAN_ARMS[arm]
    df = load_arm(spec["dataset"], spec["folds"], split, metadata_csv, dataset_tag=tag)
    if spec["agg"] == "iterations":
        df = pool_random_by_case(df)
    return df


def build_table(datasets=("AutoPET", "DeepPSMA", "DH"),
                splits=None, layout=None, margins=None,
                metadata_csv="fdg_metadata.csv", summary="mean",
                correction="fdr_bh", alpha=0.05):
    """Assemble the manuscript comparison table.

    Returns a tidy dataframe with one row per (dataset, metric, group),
    columns:
      dataset, metric, diagnosis, n, feeds, random_mean, random_sd,
      delta, p_wilcoxon, p_wilcoxon_<corr>, p_ttest,
      ni_p, ni_pass, favored

    Correction: per metric, POOLED across all datasets, on the Wilcoxon
    p-value of the SUPERIORITY family only. NI is not corrected.
    """
    splits = splits or PLAN_SPLITS
    layout = layout or PLAN_LAYOUT
    margins = margins or {"Dice": 0.05, "false_pos_vol": 5.0, "false_neg_vol": 5.0}

    rows = []
    for tag in datasets:
        sp = splits[tag]
        feeds = _load_plan_arm(tag, "feeds", sp, metadata_csv)
        rand = _load_plan_arm(tag, "random", sp, metadata_csv)
        full = _load_plan_arm(tag, "full", sp, metadata_csv)

        # groups for this dataset
        if layout[tag] == "diagnosis":
            groups = [g for g in DIAG_ORDER if g in set(feeds["diagnosis"].dropna())]
        else:
            groups = [None]  # pooled

        for col, pretty, hib, dis_only in METRICS:
            for grp in groups:
                f = feeds if grp is None else feeds[feeds["diagnosis"] == grp]
                r = rand if grp is None else rand[rand["diagnosis"] == grp]
                u = full if grp is None else full[full["diagnosis"] == grp]

                sup = paired_superiority(f, r, col, hib, dis_only, summary=summary)
                if sup is None:
                    continue  # n<2 (e.g. Negative for Dice/FNVol) -> skip row
                ni = paired_noninferiority(f, u, col, hib, dis_only,
                                           margins[col], summary=summary, alpha=alpha)
                rows.append(dict(
                    dataset=tag, metric=pretty, metric_col=col,
                    diagnosis=(DIAG_LABEL.get(grp, grp) if grp else "Pooled"),
                    n=sup["n"], feeds=sup["feeds"],
                    random_mean=sup["comp"], random_sd=sup["comp_sd"],
                    delta=sup["delta"], p_wilcoxon=sup["p_wilcoxon"],
                    p_ttest=sup["p_ttest"], favored=sup["favored"],
                    ni_p=(ni["p_noninf"] if ni else np.nan),
                    ni_pass=(ni["noninferior"] if ni else np.nan),
                ))

    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab

    # ---- correction: per metric, pooled across datasets, on Wilcoxon p ----
    corr_col = f"p_wilcoxon_{correction}"
    tab[corr_col] = np.nan
    tab["sig_" + correction] = False
    print(f"SUPERIORITY family sizes (correction denominator, method={correction}):")
    for col in tab["metric_col"].unique():
        mask = tab["metric_col"] == col
        n_tests = int(mask.sum())
        pretty = tab.loc[mask, "metric"].iloc[0]
        print(f"   {pretty}: {n_tests} tests")
        cp, rej = correct_family(tab.loc[mask, "p_wilcoxon"].values,
                                 method=correction, alpha=alpha)
        tab.loc[mask, corr_col] = cp
        tab.loc[mask, "sig_" + correction] = rej
    print(f"   SUPERIORITY total: {len(tab)} tests (NI excluded from correction)")

    return tab


# ------------------------------------------------------------------ pretty formatting
def format_table(tab, correction="fdr_bh"):
    """Render build_table() output to a display-friendly frame matching
    the manuscript figure."""
    corr_col = f"p_wilcoxon_{correction}"

    def fmt_p(x):
        if pd.isna(x):
            return "--"
        return f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"

    def fmt_val(x, metric):
        if pd.isna(x):
            return "--"
        return f"{x:.3f}" if metric == "Dice" else f"{x:.2f}"

    out = []
    for _, r in tab.iterrows():
        out.append(dict(
            Dataset=r["dataset"], Metric=r["metric"], Diagnosis=r["diagnosis"],
            FEEDS=fmt_val(r["feeds"], r["metric"]),
            Random=f"{fmt_val(r['random_mean'], r['metric'])} "
                   f"± {fmt_val(r['random_sd'], r['metric'])}",
            Δ=("+" if r["delta"] >= 0 else "") + fmt_val(r["delta"], r["metric"]),
            **{"Wilc p (raw)": fmt_p(r["p_wilcoxon"]),
               "Wilc p (FDR)": fmt_p(r[corr_col]),
               "NI vs 100%": ("✓" if r["ni_pass"] is True or r["ni_pass"] == 1 else
                              ("✗" if r["ni_pass"] is False or r["ni_pass"] == 0 else "--"))},
        ))
    return pd.DataFrame(out)
