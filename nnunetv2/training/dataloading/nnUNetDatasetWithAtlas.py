"""
nnUNetDatasetWithAtlas: subclasses nnU-Net's dataset class to load the
per-patient mean+variance atlas as additional input channels at native
resolution.

Once the atlas is loaded as extra channels in `data`, everything downstream
(random cropping, flips, rotations, elastic deformation) treats it as image
data and applies the same spatial transforms as the actual image. This is
exactly what we want: the atlas patch corresponds anatomically to the image
patch and stays aligned through augmentation.

The atlas channels are split off into `batch['atlas']` by the existing
SplitAtlasChannels transform at the very end of the pipeline, before the
network forward pass.

nnU-Net v2 dataset internals can shift between versions; this implementation
handles the refactored API where nnUNetDataset was split into:
  - nnUNetBaseDataset   (ABC, shared interface)
  - nnUNetDatasetNumpy  (loads .npz files)
  - nnUNetDatasetBlosc2 (loads .b2nd files)
...with infer_dataset_class() picking the right one per folder.

We subclass nnUNetBaseDataset and delegate __init__ to the concrete class
that infer_dataset_class() selects, so we transparently support both storage
formats.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Type

import numpy as np
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Locate the base class and concrete implementations, tolerating the two
# known nnU-Net v2 layouts.
# ---------------------------------------------------------------------------
_nnUNetBaseDataset: Type | None = None
_infer_dataset_class = None
_nnUNetDatasetNumpy = None
_nnUNetDatasetBlosc2 = None

try:
    from nnunetv2.training.dataloading.nnunet_dataset import (
        nnUNetBaseDataset as _nnUNetBaseDataset,
        nnUNetDatasetNumpy as _nnUNetDatasetNumpy,
        nnUNetDatasetBlosc2 as _nnUNetDatasetBlosc2,
        infer_dataset_class as _infer_dataset_class,
    )
except ImportError:
    pass

# Older layout: single nnUNetDataset class
if _nnUNetBaseDataset is None:
    try:
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset as _nnUNetBaseDataset  # type: ignore
    except ImportError:
        pass

if _nnUNetBaseDataset is None:
    try:
        from nnunetv2.training.dataloading.dataset import nnUNetDataset as _nnUNetBaseDataset  # type: ignore
    except ImportError:
        pass

if _nnUNetBaseDataset is None:
    raise ImportError(
        "Could not locate nnUNetBaseDataset or nnUNetDataset in "
        "nnunetv2.training.dataloading. Check your nnU-Net installation."
    )


class nnUNetDatasetWithAtlas(_nnUNetBaseDataset):
    """
    Subclass that appends per-patient atlas channels to the loaded image data.

    Because the new nnU-Net API uses two concrete storage backends
    (nnUNetDatasetNumpy for .npz, nnUNetDatasetBlosc2 for .b2nd), we
    transparently delegate to the correct one by overriding __new__ to set
    the MRO's concrete class at construction time.

    Configure via class-level attributes BEFORE instantiation:
        nnUNetDatasetWithAtlas.configure(Path('.../training_registration_to_atlas'))

    The atlas files expected per patient:
        {reg_dir}/{split}/{case_id}/registered_label.nii.gz     (mean atlas)
        {reg_dir}/{split}/{case_id}/registered_variance.nii.gz  (variance atlas)
    """

    # Class-level config; set this before nnU-Net constructs the dataset.
    atlas_reg_dir: Path | None = None
    _atlas_path_cache: dict[str, Tuple[Path, Path]] = {}

    # ------------------------------------------------------------------ #
    # Dynamic concrete-class selection                                     #
    # ------------------------------------------------------------------ #

    def __new__(cls, folder: str, case_identifiers, num_images_properties_loading_workers=0,
                **kwargs):
        """
        If infer_dataset_class is available, pick the right storage backend
        and inject it as the first concrete base so __init__ and load_case
        resolve correctly.  Falls back to plain instantiation on older nnU-Net.
        """
        if _infer_dataset_class is not None and cls is nnUNetDatasetWithAtlas:
            concrete = _infer_dataset_class(folder)
            # Build a one-off subclass whose MRO is:
            #   nnUNetDatasetWithAtlas -> <concrete backend> -> nnUNetBaseDataset
            # This makes super().load_case() in our override call the right backend.
            if concrete is not cls and not issubclass(cls, concrete):
                mixed = type(
                    cls.__name__,
                    (cls, concrete),
                    {"__new__": object.__new__},   # avoid infinite recursion
                )
                instance = object.__new__(mixed)
                return instance
        return object.__new__(cls)

    # ------------------------------------------------------------------ #
    # Atlas configuration helpers                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def configure(cls, atlas_reg_dir: Path):
        cls.atlas_reg_dir = Path(atlas_reg_dir)
        cls._atlas_path_cache = {}

    @classmethod
    def _find_atlas(cls, case_id: str) -> Tuple[Path, Path] | None:
        if case_id in cls._atlas_path_cache:
            return cls._atlas_path_cache[case_id]
        if cls.atlas_reg_dir is None:
            return None
        for split in ("train", "val", "test"):
            d = cls.atlas_reg_dir / split / case_id
            mean_p = d / "registered_label.nii.gz"
            var_p = d / "registered_variance.nii.gz"
            if mean_p.exists() and var_p.exists():
                cls._atlas_path_cache[case_id] = (mean_p, var_p)
                return mean_p, var_p
        return None

    # ------------------------------------------------------------------ #
    # Resampling utility                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resample_to_target(atlas_arr: np.ndarray, atlas_path: Path,
                            target_shape: tuple) -> np.ndarray:
        """Resample atlas to match image shape if there is any grid mismatch."""
        if atlas_arr.shape == target_shape:
            return atlas_arr
        src = sitk.GetImageFromArray(atlas_arr)
        ref = sitk.GetImageFromArray(np.zeros(target_shape, dtype=np.float32))
        resampled = sitk.Resample(src, ref, sitk.Transform(),
                                  sitk.sitkLinear, 0.0, src.GetPixelID())
        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Core override                                                        #
    # ------------------------------------------------------------------ #

    def load_case(self, key: str):
        """Override: load image+seg as usual, then append atlas channels.

        Returns (data, seg, properties) where data has shape
            (C_image + 2, Z, Y, X)
        with the trailing two channels being:
            channel C_image      = mean atlas (registered_label)
            channel C_image + 1  = variance atlas (registered_variance)
        """
        data, seg, properties = super().load_case(key)
        target_shape = data.shape[1:]   # (Z, Y, X)
        n_image_ch = data.shape[0]

        atlas_paths = self._find_atlas(key)
        if atlas_paths is None:
            # Defensive fallback: no atlas found, pad with zeros so channel
            # count stays consistent across all cases.
            mean_ch = np.zeros((1, *target_shape), dtype=data.dtype)
            var_ch  = np.zeros((1, *target_shape), dtype=data.dtype)
        else:
            mean_p, var_p = atlas_paths
            mean_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mean_p))).astype(np.float32)
            var_arr  = sitk.GetArrayFromImage(sitk.ReadImage(str(var_p))).astype(np.float32)
            var_arr  = np.clip(var_arr, 0.0, None)
            mean_arr = self._resample_to_target(mean_arr, mean_p, target_shape)
            var_arr  = self._resample_to_target(var_arr,  var_p,  target_shape)

            # Scale to [0, 1] using the empirical global max across all 597
            # training cases (0.1604). Raw atlas values are ~[0, 0.16] while
            # z-scored CT channels are ~[-3, 3]; without this the atlas signal
            # is ~50x too small to influence convolutions or loss weighting.
            _ATLAS_GLOBAL_MAX = 0.1604
            mean_arr = np.clip(mean_arr / _ATLAS_GLOBAL_MAX, 0.0, 1.0)
            var_arr  = np.clip(var_arr  / _ATLAS_GLOBAL_MAX, 0.0, 1.0)

            mean_ch  = mean_arr[None].astype(data.dtype)
            var_ch   = var_arr[None].astype(data.dtype)

        # Stash original image-channel count for SplitAtlasChannels downstream.
        properties = dict(properties) if not isinstance(properties, dict) else properties
        properties["n_image_channels"] = n_image_ch

        data = np.concatenate([data, mean_ch, var_ch], axis=0)
        return data, seg, properties