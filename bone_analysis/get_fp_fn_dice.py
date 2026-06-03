import os
import csv
import argparse
import numpy as np
import nibabel as nib
import cc3d
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def handle_arguments():
    parser = argparse.ArgumentParser(description='Process probability analysis with various options')
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    print("Using {} workers for parallel processing".format(parser.parse_args().num_workers))
    return parser.parse_args()


def nii2numpy(nii_path):
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']
    voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000
    return mask, voxel_vol


def con_comp(seg_array):
    return cc3d.connected_components(seg_array, connectivity=18)


def false_pos_pix(gt_array, pred_array):
    pred_conn_comp = con_comp(pred_array)
    false_pos, false_detected_lesion_count = 0, 0
    for idx in range(1, pred_conn_comp.max() + 1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask * gt_array).sum() == 0:
            false_pos += comp_mask.sum()
            false_detected_lesion_count += 1
    return false_pos, false_detected_lesion_count, pred_conn_comp.max()


def false_neg_pix(gt_array, pred_array):
    gt_conn_comp = con_comp(gt_array)
    false_neg, false_missed_lesion_count = 0, 0

    for idx in range(1, gt_conn_comp.max() + 1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg += comp_mask.sum()
            false_missed_lesion_count += 1

    return false_neg, false_missed_lesion_count, gt_conn_comp.max()


def dice_score(mask1, mask2):
    overlap = (mask1 * mask2).sum()
    if overlap == 0:
        return None
    return 2 * overlap / (mask1.sum() + mask2.sum())


def compute_metrics(nii_gt_path, nii_pred_path):
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, _ = nii2numpy(nii_pred_path)
    false_neg_vol, false_missed_lesion_count, num_true_lesions = false_neg_pix(gt_array, pred_array)
    false_pos_vol, false_detected_lesion_count, num_pred_lesions = false_pos_pix(gt_array, pred_array)
    return (
        dice_score(gt_array, pred_array),
        false_pos_vol * voxel_vol,
        false_neg_vol * voxel_vol,
        false_missed_lesion_count,
        false_detected_lesion_count,
        num_true_lesions, 
        num_pred_lesions
    )


def process_file(args):
    gt_file, pred_file = args
    try:
        return (os.path.basename(pred_file), *compute_metrics(gt_file, pred_file), None)
    except Exception as e:
        return (os.path.basename(pred_file), None, None, None, None, None, None, str(e))


def main():
    args = handle_arguments()

    filenames = [x for x in os.listdir(args.pred_dir) if x.endswith('.nii.gz')]
    pairs = [(os.path.join(args.gt_dir, f), os.path.join(args.pred_dir, f)) for f in filenames]

    with open(args.save_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'dice_score', 'false_pos_vol', 'false_neg_vol', 'false_missed_lesions', 'false_detected_lesions', 'num_gt_lesions', 'num_pred_lesions'])

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_file, pair): pair for pair in pairs}
            for future in tqdm.tqdm(as_completed(futures), total=len(pairs)):
                filename, dice_sc, fp_vol, fn_vol, missed, detected, num_true_lesions, num_pred_lesions, error = future.result()
                if error:
                    print(f"Error processing {filename}: {error}")
                else:
                    writer.writerow([filename, dice_sc, fp_vol, fn_vol, missed, detected, num_true_lesions, num_pred_lesions])
                    csvfile.flush()  # ensure row is written to disk immediately

if __name__ == '__main__':
    main()