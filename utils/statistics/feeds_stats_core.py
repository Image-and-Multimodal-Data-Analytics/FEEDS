"""
feeds_stats_core.py
===================================================================
Core statistics library for the FEEDS manuscript.

Extends the conventions in `stat_analysis.ipynb` and
`evaluation_with_statistics.ipynb`, with the refinements requested:

  * Random-iteration summary reported BOTH ways side by side:
      - per-case MEAN   of the 5 random models
      - per-case MEDIAN of the 5 random models
    (We do NOT pick a single median *model*; we take the per-case
     median of the metric — robust, without choosing a model.)

  * SUPERIORITY (FEEDS vs random) and NON-INFERIORITY (FEEDS vs 100%),
    paired at the case level.

  * STRATIFIED tests within groups, matching Table 3 in the paper:
      by='diagnosis', by='tracer', and by region ('organ').

  * RAW + MULTIPLICITY-CORRECTED p-values (Holm and Benjamini-Hochberg
    FDR) across groups within each metric.

  * A single DATASET SWITCH (`which=`) to run on AutoPET, DeepPSMA, DH,
    or all pooled, by changing one argument.

-------------------------------------------------------------------
mean vs median summary of the 5 random iterations
-------------------------------------------------------------------
  (a) average the 5 *predictions* voxelwise -> WRONG (a voxel-mean mask
      is a segmentation no model produced; smooths away the variance).
  (b) single model with the *median metric* -> FRAGILE (that model
      differs across datasets/metrics; the arm is not fixed).
  (c) summarize the *metric per case* across the 5 models -> USED.
      Reported as BOTH the per-case mean and the per-case median.
===================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats


TRAINER = "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres"


def _root():
    return os.environ["nnUNet_results"]


def summary_path(dataset_num, fold, split):
    return os.path.join(
        _root(), f"Dataset{dataset_num}_AutoPet", TRAINER, fold, split, "summary.json"
    )


def bone_dir(dataset_num, fold):
    return os.path.join(
        _root(), f"Dataset{dataset_num}_AutoPet", TRAINER, fold, "bone_analysis"
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


# ------------------------------------------------------------------ loader (voxel level)
def load_arm(dataset_num, folds, split, metadata_csv="fdg_metadata.csv", dataset_tag=None):
    """Per-case voxel-level dataframe for (dataset, folds, split).

    One row per (fold, case): dataset_tag, fold, fold_num, case_name,
    tracer, diagnosis, diseased, Dice, false_pos_vol, false_neg_vol.
    FN volume is NaN for non-diseased cases.
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
                fold=fold, fold_num=int(fold.split("_")[1]),
                case_name=name, tracer=_tracer(name), diagnosis=diag,
                diseased=bool(diseased), Dice=m.get("Dice", np.nan),
                false_pos_vol=fp_vol,
                false_neg_vol=(np.nan if not diseased else fn_vol),
            ))
    df = pd.DataFrame(rows)
    if not df.empty:
        neg = df["diagnosis"].eq("NEGATIVE")
        df.loc[neg, "diseased"] = False
        df.loc[neg, "false_neg_vol"] = np.nan
    return df


# ------------------------------------------------------------------ loader (region level)
PRIORITY_BONES = None  # optionally set to the notebook's PRIORITY_BONES list


def load_region_arm(dataset_num, folds, split, test_case_names,
                    metadata_csv="fdg_metadata.csv", dataset_tag=None,
                    overlap_threshold=0.4):
    """Per-case-per-organ region dataframe.

    One row per (fold, case, organ) with organ in
    {liver, lung, bone, hr_bone}. Reads *_mets.csv / *_mets_metrics.csv
    the way load_bone + organ_level do in evaluation_with_statistics.
    Region metrics (dice_sc, false_pos_vol, false_neg_vol) come from the
    *_mets_metrics.csv; the *_mets.csv gates which organs are present.
    """
    dx = _diagnosis_lookup(metadata_csv)
    keep = set(test_case_names)
    rows = []
    for fold in folds:
        bd = bone_dir(dataset_num, fold)
        if not os.path.isdir(bd):
            print(f"[skip] missing {bd}")
            continue
        for f in os.listdir(bd):
            if not f.endswith("_mets.csv"):
                continue
            case = f.replace("_mets.csv", "")
            if case not in keep:
                continue
            mets = pd.read_csv(os.path.join(bd, f))
            mfile = os.path.join(bd, f.replace("_mets.csv", "_mets_metrics.csv"))
            if not os.path.exists(mfile):
                continue
            mm = pd.read_csv(mfile)
            for col in mm.columns:
                if "vol" in col.lower() or "mm3" in col.lower():
                    mm[col] = mm[col] / 1000.0
            for col in mets.columns:
                if "Volume" in col or "mm3" in col:
                    mets[col] = mets[col] / 1000.0

            ovg = mets.get("Overlap_Volume_ground", pd.Series(0, index=mets.index)).fillna(0)
            ovp = mets.get("Overlap_Volume_pred", pd.Series(0, index=mets.index)).fillna(0)
            ov = (ovg > overlap_threshold) | (ovp > overlap_threshold)
            masks = {
                "liver": (mets["Description"] == "liver") & (ovg > 2.0),
                "lung": mets["Description"].str.contains("lung", case=False, na=False) & ov,
                "bone": mets["Type"].astype(str).str.contains("bone", case=False, na=False) & ov,
            }
            if PRIORITY_BONES is not None:
                masks["hr_bone"] = masks["bone"] & mets["Label"].isin(PRIORITY_BONES)

            diag = ("PROSTATE_CANCER" if case.lower().startswith("psma")
                    else dx.get(case))
            tr = _tracer(case)
            if "file" in mm.columns:
                mm["file"] = mm["file"].astype(str).str.replace("_0000", "", regex=False)

            for organ, mask in masks.items():
                sub = mets[mask]
                if sub.empty:
                    continue
                # region-level metrics for this organ from the mets_metrics rows
                if "Description" in mm.columns:
                    if organ in ("bone", "hr_bone"):
                        mrows = mm[mm.get("Type", "").astype(str).str.contains("bone", case=False, na=False)] \
                            if "Type" in mm.columns else mm
                    else:
                        mrows = mm[mm["Description"].astype(str).str.contains(organ, case=False, na=False)]
                    if mrows.empty:
                        mrows = mm
                else:
                    mrows = mm
                dice = mrows.get("dice_sc", pd.Series(dtype=float)).mean()
                fpv = mrows.get("false_pos_vol", pd.Series(dtype=float)).sum()
                fnv = mrows.get("false_neg_vol", pd.Series(dtype=float)).sum()
                rows.append(dict(
                    dataset_tag=dataset_tag or f"D{dataset_num}",
                    fold=fold, case_name=case, tracer=tr, diagnosis=diag,
                    diseased=True, organ=organ,
                    Dice=dice, false_pos_vol=fpv, false_neg_vol=fnv,
                ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pooling (mean AND median)
def pool_random_by_case(random_df, extra_group=None):
    """Collapse the 5 random iterations to ONE row per case (or per
    case+extra_group, e.g. 'organ'), reporting BOTH the per-case mean
    and per-case median of each metric.

    For each metric M: column M holds the mean (default arm), column
    M__median holds the median. Downstream code reads M by default and
    switches to M__median via summary='median'.
    """
    keys = ["case_name"] + ([extra_group] if extra_group else [])
    metric_cols = ["Dice", "false_pos_vol", "false_neg_vol"]
    meta_cols = [c for c in ["dataset_tag", "tracer", "diagnosis", "diseased"]
                 if c in random_df.columns]

    g = random_df.groupby(keys, dropna=False)
    out = g.agg(**{c: (c, "first") for c in meta_cols}).reset_index()
    for m in metric_cols:
        out[m] = g[m].mean().values
        out[m + "__median"] = g[m].median().values
    out["fold"] = "random_summary"
    return out


# ------------------------------------------------------------------ multiplicity correction
def _correct(pvals, method):
    """Holm or Benjamini-Hochberg FDR correction; NaNs pass through."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    corr = np.full_like(p, np.nan)
    m = int(ok.sum())
    if m == 0:
        return corr
    idx = np.where(ok)[0]
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty(m)
    if method == "holm":
        run_max = 0.0
        for i in range(m):
            run_max = max(run_max, (m - i) * ranked[i])
            adj[i] = min(run_max, 1.0)
    elif method == "fdr_bh":
        run_min = 1.0
        for i in range(m - 1, -1, -1):
            run_min = min(run_min, ranked[i] * m / (i + 1))
            adj[i] = min(run_min, 1.0)
    else:
        raise ValueError(method)
    out = np.empty(m)
    out[order] = adj
    corr[idx] = out
    return corr


# ------------------------------------------------------------------ metric spec
METRICS = [
    # (column,          pretty,   higher_is_better,  diseased_only)
    ("Dice",          "Dice",   True,   True),
    ("false_pos_vol", "FPVol",  False,  False),
    ("false_neg_vol", "FNVol",  False,  True),
]


def _colname(col, summary):
    return col if summary == "mean" else col + "__median"


def _pair(f_df, c_df, f_col, c_col, diseased_only, by=None):
    """Paired (feeds, comparator) arrays matched on case_name (+by).

    Uses distinct suffixes so f_col and c_col never collide even when
    they share a base name.
    """
    keys = ["case_name"] + ([by] if by else [])
    fcols = keys + [f_col] + (["diseased"] if "diseased" in f_df else [])
    ccols = keys + [c_col] + (["diseased"] if "diseased" in c_df else [])
    fa = f_df[fcols].rename(columns={f_col: "_f"})
    cb = c_df[ccols].rename(columns={c_col: "_c"})
    m = fa.merge(cb, on=keys, suffixes=("_ff", "_cc"))
    if diseased_only:
        for dc in ["diseased_ff", "diseased_cc", "diseased"]:
            if dc in m:
                m = m[m[dc]]
    m = m.dropna(subset=["_f", "_c"])
    return m["_f"].to_numpy(float), m["_c"].to_numpy(float)


# ------------------------------------------------------------------ superiority
def superiority_test(feeds_df, comp_df, by=None, summary="mean",
                     correction=("holm", "fdr_bh"), label="FEEDS vs random"):
    """Paired superiority tests (FEEDS vs comparator), pooled or within groups.

    by         : None (pooled), or 'diagnosis' / 'tracer' / 'organ'.
    summary    : 'mean' or 'median' random-arm column.
    correction : corrections applied ACROSS groups within each metric.
    """
    groups = [None] if by is None else sorted(
        set(feeds_df[by].dropna()) & set(comp_df[by].dropna()))
    rows = []
    for col, pretty, hib, dis_only in METRICS:
        c_col = _colname(col, summary)
        if c_col not in comp_df.columns:
            c_col = col
        for grp in groups:
            f = feeds_df if grp is None else feeds_df[feeds_df[by] == grp]
            c = comp_df if grp is None else comp_df[comp_df[by] == grp]
            a, b = _pair(f, c, col, c_col, dis_only, by=None)
            n = len(a)
            row = dict(metric=pretty, group=(grp or "ALL"), n=n)
            if n < 2:
                row["note"] = "n<2"; rows.append(row); continue
            diff = a - b
            t, p_t = stats.ttest_rel(a, b)
            try:
                _, p_w = stats.wilcoxon(a, b)
            except ValueError:
                p_w = np.nan
            row.update(feeds_mean=a.mean(), comp_val=b.mean(), delta=diff.mean(),
                       t=t, p_ttest=p_t, p_wilcoxon=p_w,
                       favored=("FEEDS" if ((diff.mean() < 0) != hib) else "comparator"))
            rows.append(row)
    df = pd.DataFrame(rows)


    # ADDING CORRECTION ACROSS GROUPS WITHIN EACH METRIC (HOLM + FDR)

    if by is not None and len(groups) > 1:
        for meth in correction:
            df[f"p_wilcoxon_{meth}"] = np.nan
            df[f"p_ttest_{meth}"] = np.nan
            for pretty in df["metric"].unique():
                mask = df["metric"] == pretty
                df.loc[mask, f"p_wilcoxon_{meth}"] = _correct(df.loc[mask, "p_wilcoxon"].values, meth)
                df.loc[mask, f"p_ttest_{meth}"] = _correct(df.loc[mask, "p_ttest"].values, meth)
    df.attrs["label"] = f"{label} [{summary}]" + (f" by {by}" if by else " pooled")
    return df


# ------------------------------------------------------------------ non-inferiority
def noninferiority_test(feeds_df, ref_df, margins, by=None, summary="mean",
                        correction=("holm", "fdr_bh"), label="FEEDS vs 100%"):
    """One-sided paired non-inferiority tests, pooled or within groups.

    HIGHER better (Dice): non-inferior if mean(FEEDS-ref) > -margin
        -> t=(md+margin)/se, p=P(T>=t).
    LOWER better (FPVol,FNVol): non-inferior if mean(FEEDS-ref) < margin
        -> t=(md-margin)/se, p=P(T<=t).
    Non-inferior when p < 0.05.
    """
    groups = [None] if by is None else sorted(
        set(feeds_df[by].dropna()) & set(ref_df[by].dropna()))
    rows = []
    for col, pretty, hib, dis_only in METRICS:
        if col not in margins:
            continue
        margin = margins[col]
        r_col = _colname(col, summary)
        if r_col not in ref_df.columns:
            r_col = col
        for grp in groups:
            f = feeds_df if grp is None else feeds_df[feeds_df[by] == grp]
            r = ref_df if grp is None else ref_df[ref_df[by] == grp]
            a, b = _pair(f, r, col, r_col, dis_only, by=None)
            n = len(a)
            row = dict(metric=pretty, group=(grp or "ALL"), n=n, margin=margin)
            if n < 2:
                row["note"] = "n<2"; rows.append(row); continue
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
            row.update(feeds_mean=a.mean(), ref_val=b.mean(), delta=md,
                       t_ni=t_stat, p_noninf=p_ni, noninferior=bool(p_ni < 0.05))
            rows.append(row)
    df = pd.DataFrame(rows)
    if by is not None and len(groups) > 1:
        for meth in correction:
            df[f"p_noninf_{meth}"] = np.nan
            for pretty in df["metric"].unique():
                mask = df["metric"] == pretty
                df.loc[mask, f"p_noninf_{meth}"] = _correct(df.loc[mask, "p_noninf"].values, meth)
                df.loc[mask, f"noninferior_{meth}"] = df.loc[mask, f"p_noninf_{meth}"] < 0.05
    df.attrs["label"] = f"{label} [{summary}]" + (f" by {by}" if by else " pooled")
    return df


# ------------------------------------------------------------------ dataset switch + driver
#
# CONFIG SCHEMA
# -------------
# `config` is a list of per-dataset dicts. Each declares the FEEDS
# reference arm and ANY NUMBER of comparator arms to test FEEDS against.
#
#   {
#     "tag":   "AutoPET",                       # dataset label
#     "feeds": (111, "fold_4"),                 # the reference arm
#     "comparators": [
#        # each comparator: a name + how to load it + which test to run
#        dict(name="random", dataset=330,
#             folds=["fold_0","fold_1","fold_2","fold_3","fold_4"],
#             kind="superiority",   agg="iterations"),   # 5 iters -> mean/median
#        dict(name="full",   dataset=111, folds="fold_9",
#             kind="noninferiority", agg="single"),      # one model
#        # add as many as you like, e.g.:
#        dict(name="dpp",     dataset=444, folds="fold_4",
#             kind="superiority",   agg="single"),
#        dict(name="kmeans",  dataset=700, folds="fold_2",
#             kind="superiority",   agg="single"),
#        dict(name="ssl_pl",  dataset=222, folds="fold_11",
#             kind="superiority",   agg="single"),
#     ],
#   }
#
# agg: "single"      -> one fold, loaded as-is.
#      "iterations"  -> multiple folds pooled per case into mean + median
#                       (use for the 5 random / any multi-seed arm).
# kind: "superiority" (paired t + Wilcoxon) or "noninferiority"
#       (one-sided; needs an entry in `margins`).
#
# For backward compatibility, the old keys still work:
#   "random":(330,[folds])  -> comparator name="random", superiority, iterations
#   "full":(111,"fold_9")   -> comparator name="full",   noninferiority, single
# ------------------------------------------------------------------

def _normalize_comparators(ds):
    """Return a list of comparator dicts, merging legacy 'random'/'full' keys."""
    comps = list(ds.get("comparators", []))
    if "random" in ds and not any(c["name"] == "random" for c in comps):
        rnum, rfolds = ds["random"]
        comps.append(dict(name="random", dataset=rnum, folds=rfolds,
                          kind="superiority", agg="iterations"))
    if "full" in ds and not any(c["name"] == "full" for c in comps):
        unum, ufold = ds["full"]
        comps.append(dict(name="full", dataset=unum, folds=ufold,
                          kind="noninferiority", agg="single"))
    return comps


def build_arms(config, split="test_predictions", metadata_csv="fdg_metadata.csv"):
    """Load the FEEDS reference and every comparator arm in `config`.

    Returns
    -------
    feeds : concatenated per-case FEEDS dataframe (across datasets).
    comparators : dict name -> {"df": per-case dataframe (mean+median
                  cols if agg=='iterations'), "kind": test kind}.
    """
    feeds_all = []
    comp_frames = {}   # name -> list of per-dataset frames
    comp_kind = {}     # name -> test kind

    for ds in config:
        tag = ds["tag"]
        fnum, ffold = ds["feeds"]
        ffolds = ffold if isinstance(ffold, list) else [ffold]
        feeds_all.append(load_arm(fnum, ffolds, split, metadata_csv, dataset_tag=tag))

        for comp in _normalize_comparators(ds):
            name = comp["name"]
            folds = comp["folds"]
            folds = folds if isinstance(folds, list) else [folds]
            raw = load_arm(comp["dataset"], folds, split, metadata_csv, dataset_tag=tag)
            if comp.get("agg", "single") == "iterations":
                raw = pool_random_by_case(raw)
            comp_frames.setdefault(name, []).append(raw)
            comp_kind[name] = comp.get("kind", "superiority")

    comparators = {
        name: dict(df=pd.concat(frames, ignore_index=True), kind=comp_kind[name])
        for name, frames in comp_frames.items()
    }
    return pd.concat(feeds_all, ignore_index=True), comparators


def select_datasets(config, which):
    """Dataset switch: 'ALL', a single tag, or a list of tags."""
    if which == "ALL":
        return config
    if isinstance(which, str):
        which = [which]
    sub = [d for d in config if d["tag"] in which]
    if not sub:
        raise ValueError(f"no datasets in config match {which}; "
                         f"available: {[d['tag'] for d in config]}")
    return sub


def run_all(config, margins, which="ALL", split="test_predictions",
            metadata_csv="fdg_metadata.csv",
            group_bys=("diagnosis", "tracer"), summaries=("mean", "median")):
    """End-to-end driver over every comparator declared in `config`.

    FEEDS is the reference; each comparator arm is tested against it with
    the test kind it declares ('superiority' or 'noninferiority'),
    pooled and stratified by each entry in `group_bys`, for each summary
    in `summaries`.

    Returns nested dict:
        results[summary][(comparator_name, kind, by)] = dataframe
    with by in {'ALL', *group_bys}.
    """
    cfg = select_datasets(config, which)
    feeds, comparators = build_arms(cfg, split, metadata_csv)

    print(f"Datasets: {[d['tag'] for d in cfg]}  |  pooled cases: {feeds['case_name'].nunique()}")
    for tag in feeds["dataset_tag"].unique():
        print(f"  {tag}: {feeds[feeds.dataset_tag == tag].case_name.nunique()}")
    print(f"Comparators: {[(n, c['kind']) for n, c in comparators.items()]}")

    results = {}
    for summ in summaries:
        results[summ] = {}
        print(f"\n{'='*70}\nRANDOM SUMMARY = {summ.upper()}\n{'='*70}")

        for name, comp in comparators.items():
            cdf, kind = comp["df"], comp["kind"]

            if kind == "noninferiority":
                if cdf.empty:
                    continue
                header = f"NON-INFERIORITY — FEEDS vs {name}"
                fn = (lambda by, _cdf=cdf, _nm=name: noninferiority_test(
                    feeds, _cdf, margins, by=by, summary=summ,
                    label=f"FEEDS vs {_nm}"))
            else:
                header = f"SUPERIORITY — FEEDS vs {name}"
                fn = (lambda by, _cdf=cdf, _nm=name: superiority_test(
                    feeds, _cdf, by=by, summary=summ,
                    label=f"FEEDS vs {_nm}"))

            print(f"\n>>> {header}  [pooled]")
            r = fn(None)
            print(r.to_string(index=False))
            results[summ][(name, kind, "ALL")] = r

            for by in group_bys:
                print(f"\n>>> {header}  [by {by}]")
                rb = fn(by)
                print(rb.to_string(index=False))
                results[summ][(name, kind, by)] = rb

    return results
