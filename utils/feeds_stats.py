"""
Paired comparison: random baseline folds (fold_0..fold_4) vs FEEDS (fold_5).

Designed to plug into stat_analysis.ipynb. Your loader
`load_dataset_folds_with_diagnosis(dataset_num=...)` returns a flat per-case
frame `all_cases` with columns:
    fold, fold_num, case_name, tracer, diagnosis,
    Dice, false_pos_vol, false_neg_vol, num_cases, diseased

Here fold_5 == FEEDS (active learning) and fold_0..fold_4 == random baselines,
all scored on the SAME test cases -> we pair on case_name.

Usage in the notebook
---------------------
    import feeds_stats

    frames = {}
    for d in [320, 333, 340]:
        _, _, _, all_cases = load_dataset_folds_with_diagnosis(
            dataset_num=d,
            folds=[f'fold_{i}' for i in range(0, 6)],   # 0..5 is enough
            validation="test_predictions",             # or your validation tag
        )
        frames[d] = all_cases

    results = feeds_stats.run_many(frames)   # dict {dataset: results_df}
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# metric -> direction of the FEEDS hypothesis
#   "greater": FEEDS expected HIGHER (Dice)
#   "less":    FEEDS expected LOWER  (FP / FN volume)
METRICS = {
    "Dice":          "greater",
    "false_pos_vol": "less",
    "false_neg_vol": "less",
}

FEEDS_FOLD = 5
RAND_FOLDS = [0, 1, 2, 3, 4]


def _per_case_table(all_cases: pd.DataFrame):
    """Return (random_mean_per_case, feeds_per_case) merged on case_name."""
    df = all_cases.copy()

    rand = (df[df["fold_num"].isin(RAND_FOLDS)]
              .groupby("case_name")[list(METRICS)]
              .mean())
    feeds = (df[df["fold_num"] == FEEDS_FOLD]
               .groupby("case_name")[list(METRICS)]
               .mean())

    merged = rand.join(feeds, lsuffix="_rand", rsuffix="_feeds", how="inner")
    return merged


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return (np.nan, np.nan)
    boot = np.array([rng.choice(diff, diff.size, replace=True).mean()
                     for _ in range(n_boot)])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def run(all_cases: pd.DataFrame, dataset_label="") -> pd.DataFrame:
    """Run paired t-test + Wilcoxon for one dataset's all_cases frame."""
    merged = _per_case_table(all_cases)
    if dataset_label:
        print(f"\n=== Dataset {dataset_label} === paired on {len(merged)} cases")

    rows = []
    for metric, side in METRICS.items():
        r = merged[f"{metric}_rand"].to_numpy()
        f = merged[f"{metric}_feeds"].to_numpy()
        mask = ~(np.isnan(r) | np.isnan(f))
        r, f = r[mask], f[mask]
        if r.size < 3:
            print(f"  [skip] {metric}: only {r.size} valid pairs")
            continue

        diff = f - r  # FEEDS minus random

        t_stat, t_p_two = ttest_rel(f, r)
        if side == "greater":
            t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
            w_stat, w_p = wilcoxon(f, r, alternative="greater")
        else:
            t_p = t_p_two / 2 if t_stat < 0 else 1 - t_p_two / 2
            w_stat, w_p = wilcoxon(f, r, alternative="less")

        lo, hi = bootstrap_ci(diff)
        rows.append({
            "dataset": dataset_label,
            "metric": metric,
            "n_pairs": int(mask.sum()),
            "mean_rand": r.mean(),
            "mean_feeds": f.mean(),
            "mean_diff(feeds-rand)": diff.mean(),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "t_p_1sided": t_p,
            "wilcoxon_p_1sided": w_p,
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        pd.set_option("display.float_format", lambda x: f"{x:.4g}")
        print(res.to_string(index=False))
    return res


def run_many(frames: dict) -> dict:
    """frames: {dataset_num: all_cases_df}. Returns {dataset_num: results_df}."""
    out = {}
    for d, ac in frames.items():
        out[d] = run(ac, dataset_label=str(d))
    return out