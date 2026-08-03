# tta_uncertainty_predictor.py

import os
import multiprocessing
from time import sleep
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, save_json, load_json
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.inference.sliding_window_prediction import compute_gaussian


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-8  # Same as your ensemble script


# ---------------------------------------------------------------------------
# Entropy — matches your ensemble script exactly
# ---------------------------------------------------------------------------

def compute_entropy(prob: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Matches:
        def compute_entropy(prob, axis=0):
            prob = np.clip(prob, epsilon, 1.0)
            return -np.sum(prob * np.log(prob), axis=axis)
    """
    prob = np.clip(prob, EPSILON, 1.0)
    return -np.sum(prob * np.log(prob), axis=axis)


# ---------------------------------------------------------------------------
# Uncertainty decomposition — matches your ensemble script exactly
#
# Your ensemble script:
#   aleatoric = mean over members of per-member entropy     (avg of H[p_i])
#   total     = entropy of the mean probability             (H[avg p_i])
#   epistemic = total - aleatoric                           (mutual information)
# ---------------------------------------------------------------------------

def compute_uncertainty_maps(
    all_probs: np.ndarray,   # [N_tta, num_classes, X, Y, Z]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (aleatoric, epistemic, total), each of shape [X, Y, Z].

    Mirrors your ensemble script:
        aleatoric_uncertainty = np.mean(entropy_list, axis=0)
        total_uncertainty     = compute_entropy(avg_prob, axis=0)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    """
    # Per-TTA-member entropy → averaged  (aleatoric)
    entropy_list = [compute_entropy(p, axis=0) for p in all_probs]   # list of [X,Y,Z]
    aleatoric = np.mean(entropy_list, axis=0)                         # [X, Y, Z]

    # Entropy of the mean probability map  (total)
    avg_prob = np.mean(all_probs, axis=0)                             # [C, X, Y, Z]
    total = compute_entropy(avg_prob, axis=0)                         # [X, Y, Z]

    # Epistemic = mutual information
    epistemic = total - aleatoric                                     # [X, Y, Z]

    return aleatoric, epistemic, total


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def apply_flip(x: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
    """Flip along spatial axes (0-indexed spatial → dim+1 for 4D [C,X,Y,Z])."""
    for ax in axes:
        x = torch.flip(x, [ax + 1])
    return x

def undo_flip(x: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
    return apply_flip(x, axes)   # flipping is its own inverse

def apply_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur per channel via scipy (handles arbitrary 3D volumes)."""
    arr = x.cpu().numpy()
    blurred = np.stack(
        [gaussian_filter(arr[c], sigma=sigma) for c in range(arr.shape[0])],
        axis=0,
    )
    return torch.from_numpy(blurred).to(x.device).to(x.dtype)


def build_tta_augmentations(
    flip_axes: Tuple[Tuple[int, ...], ...],
    gaussian_sigmas: Tuple[float, ...],
):
    """
    Returns a list of (augment_fn, deaugment_fn) pairs.
    Identity is always included as the first entry.
    """
    augmentations = []

    # Identity
    augmentations.append((lambda x: x, lambda x: x))

    # Flips (spatially invertible)
    for axes in flip_axes:
        fwd = (lambda ax: lambda x: apply_flip(x, ax))(axes)
        inv = (lambda ax: lambda x: undo_flip(x, ax))(axes)
        augmentations.append((fwd, inv))

    # Gaussian blurs (no spatial inversion needed)
    for sigma in gaussian_sigmas:
        fwd = (lambda s: lambda x: apply_gaussian_blur(x, s))(sigma)
        augmentations.append((fwd, lambda x: x))

    return augmentations


# ---------------------------------------------------------------------------
# Save uncertainty maps — matches your ensemble script's save logic exactly
# ---------------------------------------------------------------------------

def save_uncertainty_maps(
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    total: np.ndarray,
    ref_img: sitk.Image,
    case_id: str,
    uncer_dir: str,
):
    """
    Normalises each map to [0, 1] and saves as .nii.gz.
    Mirrors your ensemble script's Step 10 exactly.
    """
    for name, array in zip(
        ["aleatoric", "epistemic", "total"],
        [aleatoric, epistemic, total],
    ):
        arr_min, arr_max = np.min(array), np.max(array)
        if arr_max > arr_min:
            arr_norm = (array - arr_min) / (arr_max - arr_min)
        else:
            arr_norm = np.zeros_like(array)

        unc_img = sitk.GetImageFromArray(arr_norm.astype(np.float32))
        unc_img.CopyInformation(ref_img)

        out_path = os.path.join(uncer_dir, f"{case_id}_{name}_uncertainty.nii.gz")
        sitk.WriteImage(unc_img, out_path)
        print(f'  [TTA] Saved {name} uncertainty → {os.path.basename(out_path)}')


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class TTAUncertaintyPredictor(nnUNetPredictor):
    """
    Extends nnUNetPredictor with:
      - Custom TTA (flips + Gaussian blurs)
      - Entropy-based uncertainty decomposition matching the ensemble script:
            aleatoric = mean H[p_i]          (avg per-augmentation entropy)
            total     = H[mean p_i]          (entropy of the mean)
            epistemic = total - aleatoric    (mutual information / model uncertainty)
      - Saves aleatoric / epistemic / total uncertainty maps as .nii.gz
    """

    def __init__(
        self,
        flip_axes: Tuple[Tuple[int, ...], ...] = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)),
        gaussian_sigmas: Tuple[float, ...] = (0.5, 1.0),
        uncer_dir: str = None,
        **kwargs,
    ):
        """
        flip_axes:       Spatial axes to flip (0=x, 1=y, 2=z for 3D images).
        gaussian_sigmas: Sigmas for Gaussian blur augmentations.
        uncer_dir:       Where to save uncertainty .nii.gz maps.
                         If None, maps are not saved (returned only).
        kwargs:          Passed straight to nnUNetPredictor
                         (tile_step_size, device, verbose, etc.)
        """
        super().__init__(**kwargs)
        self.tta_augmentations = build_tta_augmentations(flip_axes, gaussian_sigmas)
        self.uncer_dir = uncer_dir
        if uncer_dir is not None:
            os.makedirs(uncer_dir, exist_ok=True)

        print(
            f'[TTA] {len(self.tta_augmentations)} augmentations total: '
            f'1 identity + {len(flip_axes)} flips + {len(gaussian_sigmas)} blurs'
        )

    # ------------------------------------------------------------------
    # TTA forward pass — collects per-augmentation probability maps
    # ------------------------------------------------------------------

    def _predict_tta_all_probs(self, data: torch.Tensor) -> np.ndarray:
        """
        Runs all TTA augmentations and returns stacked probabilities.

        Returns:
            all_probs: np.ndarray [N_tta, num_classes, X, Y, Z]
                       (softmax probabilities, de-augmented)
        """
        all_probs = []

        for aug_fn, deaug_fn in self.tta_augmentations:
            # 1. Augment input
            aug_data = aug_fn(data)

            # 2. Sliding-window prediction → logits [C, X, Y, Z]
            logits = self.predict_sliding_window_return_logits(aug_data)

            # 3. Softmax → probabilities
            probs = torch.softmax(logits.float(), dim=0)   # [C, X, Y, Z]

            # 4. De-augment (undo spatial transforms on the output)
            probs = deaug_fn(probs)

            all_probs.append(probs.cpu().numpy())

        return np.stack(all_probs, axis=0)   # [N_tta, C, X, Y, Z]

    # ------------------------------------------------------------------
    # Override predict_from_data_iterator
    # ------------------------------------------------------------------

    def predict_from_data_iterator(
        self,
        data_iterator,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = 2,
    ):
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            # Collect uncertainty data to save after export pool finishes
            pending_uncertainty = []   # list of (ofile, case_id, all_probs, ref_nii_path)

            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile   = preprocessed['ofile']
                properties = preprocessed['data_properties']

                if ofile is not None:
                    print(f'\n[TTA] Predicting {os.path.basename(ofile)}')
                else:
                    print(f'\n[TTA] Predicting image of shape {data.shape}')

                # Throttle if export workers are falling behind
                proceed = not check_workers_alive_and_busy(
                    export_pool, worker_list, r, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        export_pool, worker_list, r, allowed_num_queued=2
                    )

                # ── TTA inference ──────────────────────────────────────
                # all_probs: [N_tta, C, X, Y, Z]  numpy, matches ensemble script shape
                all_probs = self._predict_tta_all_probs(data)

                # Mean probability map — same as avg_prob in ensemble script
                avg_prob = np.mean(all_probs, axis=0)              # [C, X, Y, Z]

                # Pseudo-logits for nnUNet export (argmax-equivalent to avg_prob)
                pseudo_logits = torch.from_numpy(
                    np.log(np.clip(avg_prob, EPSILON, 1.0))
                ).cpu()

                # ── Export segmentation ────────────────────────────────
                if ofile is not None:
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((pseudo_logits, properties, self.configuration_manager,
                              self.plans_manager, self.dataset_json, ofile,
                              save_probabilities),)
                        )
                    )
                    if self.uncer_dir is not None:
                        case_id = os.path.basename(ofile)
                        # Strip the file ending so it matches your naming convention
                        for ending in ['.nii.gz', '.nii', '.mha', '.nrrd']:
                            if case_id.endswith(ending):
                                case_id = case_id[: -len(ending)]
                                break
                        ref_nii_path = ofile + self.dataset_json['file_ending']
                        pending_uncertainty.append((ofile, case_id, all_probs, ref_nii_path))
                else:
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            ((pseudo_logits, self.plans_manager, self.configuration_manager,
                              self.label_manager, properties, save_probabilities),)
                        )
                    )

                if ofile is not None:
                    print(f'  [TTA] Queued export for {os.path.basename(ofile)}')

            ret = [i.get()[0] for i in r]

        # ── Save uncertainty maps after all exports are done ──────────
        # We wait until here so the .nii.gz segmentation files exist
        # and we can use them as the SimpleITK reference image.
        if self.uncer_dir is not None:
            for ofile, case_id, all_probs, ref_nii_path in pending_uncertainty:
                # Compute the three uncertainty maps — same math as ensemble script
                aleatoric, epistemic, total = compute_uncertainty_maps(all_probs)

                # Load the exported segmentation as the reference image
                # (correct spacing / origin / direction after nnUNet resampling)
                seg_path = ofile + self.dataset_json['file_ending']
                if os.path.isfile(seg_path):
                    ref_img = sitk.ReadImage(seg_path)
                else:
                    # Fallback: uncertainty will be in preprocessed space
                    print(f'  [TTA] WARNING: reference .nii.gz not found at {seg_path}, '
                          f'saving uncertainty in preprocessed space.')
                    ref_img = None

                if ref_img is not None:
                    save_uncertainty_maps(
                        aleatoric, epistemic, total,
                        ref_img, case_id, self.uncer_dir,
                    )

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

    # ------------------------------------------------------------------
    # Convenience entry point — mirrors predict_from_files interface
    # ------------------------------------------------------------------

    def predict_from_files_with_uncertainty(
        self,
        input_folder: str,
        output_folder: str,
        overwrite: bool = True,
        num_processes_preprocessing: int = 2,
        num_processes_segmentation_export: int = 2,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
    ):
        maybe_mkdir_p(output_folder)

        save_json(
            {
                'input_folder': input_folder,
                'output_folder': output_folder,
                'uncer_dir': self.uncer_dir,
                'n_tta_augmentations': len(self.tta_augmentations),
            },
            join(output_folder, 'tta_predict_args.json'),
        )
        save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)

        list_of_lists, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(
                input_folder, output_folder,
                folder_with_segs_from_prev_stage,
                overwrite, part_id, num_parts,
                save_probabilities=False,
            )

        if len(list_of_lists) == 0:
            print('[TTA] Nothing to predict.')
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists, seg_from_prev_stage_files,
            output_filename_truncated, num_processes_preprocessing,
        )

        return self.predict_from_data_iterator(
            data_iterator,
            save_probabilities=False,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )s