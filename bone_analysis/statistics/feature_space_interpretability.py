"""
feature_space_interpretability.py
===================================================================
"Feature embeddings interpretability" analysis requested in the thread:

    "add a section of what the neighboring and farthest away cases
     typically are in the feature space ... give some more examples of
     analysis of farthest away points from the fixed 10% pool. Maybe it
     is picking up cases not as well represented in the 10%? ... show
     that FEEDS is indeed picking up diversity."

This recomputes, from the DINOv2 embeddings, exactly the quantity that
FEEDS ranks on (Eq. 2 in the manuscript):

    d_j = 1 - max_{i in L_t}  cos(z_j, z_i)

i.e. for each unlabeled case j, its cosine distance to its NEAREST
labeled (fixed-10%) case, computed WITHIN each tracer group.

It then produces the evidence for the results subsection:

  (A) Nearest neighbours: for the smallest-d_j cases, report the
      labeled case each is closest to. Longitudinal scans of the same
      patient are flagged automatically (same patient id, different
      scan date) — this reproduces the observation that the two
      closest points are often longitudinal scans of one patient.

  (B) Farthest cases: for the largest-d_j cases (the ones FEEDS
      selects first), tabulate their diagnosis/tracer mix and compare
      it to the diagnosis/tracer mix of the fixed 10% pool, to test
      the hypothesis "FEEDS picks up cases under-represented in 10%".

  (C) A representation-ratio table: for each (tracer, diagnosis) group,
      share in the fixed-10% pool vs share among the FEEDS-selected
      farthest X%. A ratio > 1 means FEEDS over-samples that group
      relative to the seed, which is the diversity claim.

Run this once per dataset that has embeddings on disk (AutoPET, and
DeepPSMA if embeddings were extracted for it), as Indrani suggests
"maybe we can find some patterns in the autopet too".

-------------------------------------------------------------------
INPUTS (point these at your embedding files)
-------------------------------------------------------------------
Expected: a table with one row per case and columns
    case_name, tracer, diagnosis, split ('labeled'/'unlabeled'),
    and either
      - an embedding vector stored as a .npy sidecar keyed by
        case_name, OR
      - 768 columns emb_0..emb_767 in the same table.

Adjust load_embeddings() to your actual file; the analysis functions
below are agnostic to how they were loaded.
===================================================================
"""

import os
import re
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ loading
def load_embeddings(emb_table_csv, emb_npy=None):
    """Load embeddings into (meta_df, Z) where Z is (N, 768) L2-normalizable.

    Two layouts supported:
      1. emb_table_csv has emb_0..emb_767 columns  -> emb_npy=None
      2. emb_table_csv has metadata only, vectors in emb_npy aligned by row
    """
    meta = pd.read_csv(emb_table_csv)
    emb_cols = [c for c in meta.columns if re.fullmatch(r"emb_\d+", str(c))]
    if emb_cols:
        Z = meta[emb_cols].to_numpy(float)
        meta = meta.drop(columns=emb_cols)
    elif emb_npy:
        Z = np.load(emb_npy)
        assert len(Z) == len(meta), "emb_npy rows must align with emb_table rows"
    else:
        raise ValueError("No emb_* columns and no emb_npy provided.")
    required = {"case_name", "tracer", "diagnosis", "split"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"embedding table missing columns: {missing}")
    return meta, Z


def _l2norm(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


# ------------------------------------------------------------------ patient id
def _patient_id(case_name):
    """Extract the patient id (strip the scan-date/scan-folder suffix).

    fdg_0011f3deaf_04-04-2003-... -> fdg_0011f3deaf
    psma_fad59ccacf4b88f2_2021-11-06 -> psma_fad59ccacf4b88f2
    """
    parts = case_name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_name


# ------------------------------------------------------------------ Eq.2 distances
def nearest_labeled(meta, Z):
    """For every unlabeled case, its cosine distance d_j to the nearest
    labeled case, WITHIN tracer (Eq. 2). Returns an augmented dataframe
    for the unlabeled rows with columns: d_j, nn_case, nn_diagnosis,
    is_longitudinal_of_nn.
    """
    Zn = _l2norm(Z)
    meta = meta.reset_index(drop=True)
    out = []
    for tracer in meta["tracer"].dropna().unique():
        idx_t = meta.index[meta["tracer"] == tracer].to_numpy()
        lab = idx_t[meta.loc[idx_t, "split"] == "labeled"]
        unl = idx_t[meta.loc[idx_t, "split"] == "unlabeled"]
        if len(lab) == 0 or len(unl) == 0:
            continue
        sims = Zn[unl] @ Zn[lab].T           # (n_unl, n_lab) cosine sims
        nn_pos = sims.argmax(axis=1)
        nn_sim = sims.max(axis=1)
        for k, j in enumerate(unl):
            nn_j = lab[nn_pos[k]]
            uc = meta.loc[j, "case_name"]
            nc = meta.loc[nn_j, "case_name"]
            out.append(
                dict(
                    case_name=uc,
                    tracer=tracer,
                    diagnosis=meta.loc[j, "diagnosis"],
                    d_j=1.0 - nn_sim[k],
                    nn_case=nc,
                    nn_diagnosis=meta.loc[nn_j, "diagnosis"],
                    is_longitudinal_of_nn=(_patient_id(uc) == _patient_id(nc)),
                )
            )
    return pd.DataFrame(out).sort_values("d_j").reset_index(drop=True)


# ------------------------------------------------------------------ (A) nearest
def nearest_examples(dist_df, k=15):
    """The k smallest-d_j (most redundant) cases and what they duplicate.

    Reproduces the 'two closest points are longitudinal scans of the
    same patient' observation and quantifies how common that is.
    """
    head = dist_df.nsmallest(k, "d_j").copy()
    frac_long = dist_df.assign(
        rank=dist_df["d_j"].rank(method="first")
    )
    # among the closest 10%, how many are longitudinal duplicates?
    n_close = max(1, int(0.10 * len(dist_df)))
    closest = dist_df.nsmallest(n_close, "d_j")
    share = closest["is_longitudinal_of_nn"].mean()
    print(f"Among the closest {n_close} unlabeled cases (smallest d_j), "
          f"{share:.0%} are longitudinal scans of their nearest labeled case.")
    return head


# ------------------------------------------------------------------ (B) farthest
def farthest_examples(dist_df, k=15):
    """The k largest-d_j cases — the ones FEEDS annotates first."""
    return dist_df.nlargest(k, "d_j").copy()


# ------------------------------------------------------------------ (C) diversity
def representation_ratio(meta, dist_df, select_pct=0.20):
    """Compare group shares in the fixed-10% seed vs the FEEDS-selected
    farthest X%. ratio>1 => FEEDS over-samples that group vs the seed.
    """
    seed = meta[meta["split"] == "labeled"]
    n_sel = int(round(select_pct * (meta["split"] == "unlabeled").sum()))
    selected = dist_df.nlargest(n_sel, "d_j")

    def shares(df):
        g = df.groupby(["tracer", "diagnosis"]).size()
        return (g / g.sum()).rename("share")

    seed_sh = shares(seed)
    sel_sh = shares(selected)
    tab = pd.concat(
        [seed_sh.rename("seed_10pct_share"),
         sel_sh.rename("feeds_selected_share")],
        axis=1,
    ).fillna(0.0)
    tab["ratio_selected_over_seed"] = np.where(
        tab["seed_10pct_share"] > 0,
        tab["feeds_selected_share"] / tab["seed_10pct_share"],
        np.inf,
    )
    return tab.sort_values("ratio_selected_over_seed", ascending=False)


# ------------------------------------------------------------------ driver
def run(emb_table_csv, emb_npy=None, dataset_tag="AutoPET",
        select_pct=0.20, k=15, out_dir="."):
    print(f"\n===== Feature-space interpretability: {dataset_tag} =====")
    meta, Z = load_embeddings(emb_table_csv, emb_npy)
    dist = nearest_labeled(meta, Z)

    print("\n(A) Nearest (most redundant) cases:")
    near = nearest_examples(dist, k=k)
    print(near.to_string(index=False))

    print("\n(B) Farthest cases (FEEDS annotates these first):")
    far = farthest_examples(dist, k=k)
    print(far.to_string(index=False))

    print(f"\n(C) Diversity — group share seed-10% vs FEEDS-selected "
          f"farthest {select_pct:.0%}:")
    ratio = representation_ratio(meta, dist, select_pct=select_pct)
    print(ratio.round(3).to_string())

    os.makedirs(out_dir, exist_ok=True)
    dist.to_csv(os.path.join(out_dir, f"{dataset_tag}_case_distances.csv"), index=False)
    near.to_csv(os.path.join(out_dir, f"{dataset_tag}_nearest_examples.csv"), index=False)
    far.to_csv(os.path.join(out_dir, f"{dataset_tag}_farthest_examples.csv"), index=False)
    ratio.to_csv(os.path.join(out_dir, f"{dataset_tag}_representation_ratio.csv"))
    print(f"\nSaved 4 CSVs to {out_dir}")
    return dict(dist=dist, near=near, far=far, ratio=ratio)


if __name__ == "__main__":
    # Example — edit paths to your embedding export:
    # run("dinov2_embeddings_autopet.csv", dataset_tag="AutoPET")
    # run("dinov2_embeddings_deeppsma.csv", dataset_tag="DeepPSMA")
    print(__doc__)
