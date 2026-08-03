# main.py  – unified entry point (replaces probability_analysis.py + main_mets.py)
"""
Usage examples
--------------
# lesion analysis
python main.py --mode lesion -f comb -i /path/images -ld /path/labels \
               -b /path/bone -p /path/preds

# bone-metastasis analysis
python main.py --mode mets -f fold_0 -i /path/images -ld /path/labels \
               -b /path/bone -p /path/preds

# both
python main.py --mode both ...
"""

import os
import glob
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

from process import process_image, process_bone_mets


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Lesion / bone-metastasis analysis")
    p.add_argument("--mode", choices=["lesion", "mets", "both"], default="both")
    p.add_argument("-k",  "--key",          default="*")
    p.add_argument("-f",  "--fold",         default="comb")
    p.add_argument("-j",  "--workers",      type=int, default=4,
                   help="Parallel worker processes (default: 4)")
    p.add_argument("-s",  "--skip-existing",action="store_true")
    p.add_argument("-i",  "--image-dir",    required=True)
    p.add_argument("-ld", "--label-dir",    required=True)
    p.add_argument("-b",  "--bone-dir",     required=True)
    p.add_argument("-p",  "--pred-dir",     required=True)
    p.add_argument("-u",  "--uncertainty-dir", default=None)
    p.add_argument("-o",  "--save-dir",     default="./results")
    p.add_argument("--organ_map",           default="./ts_table.csv")
    return p


# ── worker functions (must be top-level for multiprocessing) ─────────────────

def _lesion_worker(lesion_file, dirs, only_lesion):
    try:
        return process_image(lesion_file, dirs, only_lesion)
    except Exception as e:
        print(f"ERROR in lesion worker for {lesion_file}: {e}")
        return None


def _mets_worker(lesion_file, dirs, save_dir, skip_existing):
    import os
    image_id = os.path.basename(lesion_file).split(".")[0]
    out_csv  = os.path.join(save_dir, f"{image_id}_mets.csv")

    if skip_existing and os.path.exists(out_csv):
        return f"skipped:{image_id}"

    try:
        obj = process_bone_mets(lesion_file, dirs)
        if obj is None:
            return f"missing:{image_id}"
        obj.merged_table.to_csv(out_csv, index=False)
        obj.metrics.to_csv(
            os.path.join(save_dir, f"{image_id}_mets_metrics.csv"), index=False
        )
        return f"ok:{image_id}"
    except Exception as e:
        print(f"ERROR in mets worker for {image_id}: {e}", flush=True)
        return f"error:{image_id}"


# ── orchestration ─────────────────────────────────────────────────────────────

def run_lesion_analysis(lesion_files, dirs, args, save_dir):
    """Process lesion files in parallel, periodically checkpoint to pickle."""
    save_file = os.path.join(save_dir, f"summary_{args.key.replace('*','')}.pkl")

    # load checkpoint
    existing_summary = []
    already_done     = set()
    if os.path.exists(save_file):
        with open(save_file, "rb") as f:
            existing_summary = pickle.load(f)
        already_done = {d["id"] for d in existing_summary
                        if isinstance(d, dict) and "id" in d}
        print(f"Resuming: {len(already_done)} already processed")

    if args.skip_existing:
        lesion_files = [lf for lf in lesion_files
                        if os.path.basename(lf).split(".")[0] not in already_done]
        print(f"{len(lesion_files)} files remaining after skip-existing filter")

    worker = partial(_lesion_worker, dirs=dirs, only_lesion=False)

    counter = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, lf): lf for lf in lesion_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Lesion analysis"):
            result = fut.result()
            if result is None:
                continue
            pred_stats, ground_stats, image_stats = result
            existing_summary.append({
                "id":           image_stats["image_id"],
                "pred_stats":   pred_stats,
                "ground_stats": ground_stats,
                "image_stats":  image_stats,
            })
            counter += 1
            if counter % 10 == 0:          # checkpoint every 10 files
                _save_pkl(existing_summary, save_file)
                print(f"Checkpoint: {counter} processed", flush=True)

    _save_pkl(existing_summary, save_file)
    print(f"Done. Processed {counter} images. Summary → {save_file}")


def run_mets_analysis(lesion_files, dirs, args, save_dir):
    worker = partial(_mets_worker, dirs=dirs,
                     save_dir=save_dir, skip_existing=args.skip_existing)

    results = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, lf): lf for lf in lesion_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Mets analysis"):
            tag = (fut.result() or "error:?").split(":")[0]
            results[tag] = results.get(tag, 0) + 1

    print(f"Mets done – {results}")


def _save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    dirs = {
        "images":      args.image_dir,
        "labels":      args.label_dir,
        "bone":        args.bone_dir,
        "pred":        args.pred_dir,
        "uncertainty": args.uncertainty_dir,
    }

    save_dir = os.path.join(args.save_dir, "bone_analysis")
    os.makedirs(save_dir, exist_ok=True)

    lesion_files = sorted(
        glob.glob(os.path.join(dirs["labels"], f"{args.key}*.nii.gz"))
    )
    print(f"Found {len(lesion_files)} label files")

    if args.mode in ("lesion", "both"):
        run_lesion_analysis(lesion_files, dirs, args, save_dir)

    if args.mode in ("mets", "both"):
        run_mets_analysis(lesion_files, dirs, args, save_dir)


if __name__ == "__main__":
    main()