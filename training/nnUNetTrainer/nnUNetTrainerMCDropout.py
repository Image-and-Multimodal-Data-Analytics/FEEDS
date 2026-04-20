import torch
import torch.nn as nn
from torch import autocast
import numpy as np
from typing import Union, Tuple, List
import os
import nibabel as nib

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager
)
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset_multitask import nnUNetDatasetMultiTask
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.custom_transforms import Misalign2
from training.nnUNetTrainer import autoPET3_Trainer


# ─────────────────────────────────────────────
# Utility: recursively enable dropout at inference
# ─────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module) -> None:
    """Force all Dropout layers into training mode (BN stays in eval)."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


# ─────────────────────────────────────────────
# Dropout injection helper
# ─────────────────────────────────────────────

def inject_dropout(
    module: nn.Module,
    dropout_p: float,
    use_3d: bool = True
) -> nn.Module:
    """
    Recursively walk a network and insert a Dropout layer
    after every Conv → Norm → nonlinearity block (identified
    by the presence of an InstanceNorm or BatchNorm child).

    Works by replacing each nn.Sequential block with a version
    that has a Dropout appended.

    Args:
        module:     root module to modify in-place
        dropout_p:  dropout probability
        use_3d:     use Dropout3d for volumetric data, else Dropout2d

    Returns:
        The modified module (also edited in-place).
    """
    DropoutClass = nn.Dropout3d if use_3d else nn.Dropout2d

    for name, child in module.named_children():
        # Recurse first so we work bottom-up
        inject_dropout(child, dropout_p, use_3d)

        # Target nn.Sequential blocks that contain a normalisation layer
        # — these are the Conv-Norm-ReLU stacks inside nnU-Net
        if isinstance(child, nn.Sequential):
            has_norm = any(
                isinstance(m, (nn.InstanceNorm2d,
                               nn.InstanceNorm3d,
                               nn.BatchNorm2d,
                               nn.BatchNorm3d))
                for m in child.modules()
            )
            already_has_dropout = any(
                isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                for m in child.modules()
            )
            if has_norm and not already_has_dropout:
                new_seq = nn.Sequential(
                    *list(child.children()),
                    DropoutClass(p=dropout_p)
                )
                setattr(module, name, new_seq)

    return module


# ─────────────────────────────────────────────
# MC Dropout inference helper
# ─────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run N stochastic forward passes and aggregate results.

    Args:
        model:    network with dropout layers
        x:        input tensor (B, C, ...)
        n_passes: number of MC samples

    Returns:
        mean_prob   – mean softmax probability  (B, C, ...)
        variance    – per-voxel variance         (B, C, ...)
        pred_entropy– predictive entropy         (B, ...)
    """
    model.eval()
    enable_mc_dropout(model)

    samples = []
    for _ in range(n_passes):
        logits = model(x)
        # nnU-Net returns a list when deep supervision is on
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        samples.append(probs)

    # Stack → (N, B, C, ...)
    samples = torch.stack(samples, dim=0)

    mean_prob = samples.mean(dim=0)                       # (B, C, ...)
    variance  = samples.var(dim=0)                        # (B, C, ...)

    eps = 1e-8
    pred_entropy = -(mean_prob * torch.log(mean_prob + eps)).sum(dim=1)
    # (B, ...)

    return mean_prob, variance, pred_entropy


# ─────────────────────────────────────────────
# The trainer
# ─────────────────────────────────────────────

class nnUNetTrainerMCDropout(autoPET3_Trainer):
    """
    nnU-Net trainer that injects spatial Dropout into every
    Conv-Norm-Act block and trains with it enabled.

    At inference time call `predict_with_uncertainty()` to obtain
    per-voxel mean predictions, variance, and predictive entropy.

    Hyper-parameters
    ----------------
    dropout_p  : float  – dropout probability (default 0.1)
    n_mc_passes: int    – forward passes at inference (default 20)

    Usage
    -----
    Train exactly like any other nnU-Net trainer:

        nnUNetv2_train DATASET_ID 3d_fullres FOLD \\
            -tr nnUNetTrainerMCDropout

    Then predict with uncertainty (see predict_with_uncertainty).
    """

    # ── class-level hyper-parameters (override by subclassing) ──────────
    dropout_p:   float = 0.1
    n_mc_passes: int   = 20
    # ────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )
        self.print_to_log_file(
            f"nnUNetTrainerMCDropout | "
            f"dropout_p={self.dropout_p} | "
            f"n_mc_passes={self.n_mc_passes}"
        )

    # ── network building ─────────────────────────────────────────────────

    def build_network_architecture(
        self,
        plans_manager:        PlansManager,
        dataset_json:         dict,
        configuration_manager: ConfigurationManager,
        num_input_channels:   int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        Build the standard nnU-Net architecture then inject dropout
        into every Conv-Norm-Act Sequential block.
        """
        network = super().build_network_architecture(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision,
        )

        # Determine spatial dimensionality from the configuration
        patch_size = configuration_manager.patch_size
        use_3d = len(patch_size) == 3

        network = inject_dropout(network, self.dropout_p, use_3d=use_3d)

        # Count injected layers so we can log it
        n_drop = sum(
            1 for m in network.modules()
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
        )
        self.print_to_log_file(
            f"Injected {n_drop} dropout layers (p={self.dropout_p})"
        )

        return network

    # ── training step ────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        Standard nnU-Net train step.
        Dropout is automatically active because the network is in
        .train() mode — nothing extra is needed here.
        """
        return super().train_step(batch)

    # ── validation step ──────────────────────────────────────────────────

    def validation_step(self, batch: dict) -> dict:
        """
        Run a *single* deterministic forward pass for validation
        (dropout disabled) so that validation metrics are comparable
        to a standard nnU-Net run.
        """
        # super() calls self.network.eval() internally via
        # on_validation_epoch_start, so dropout is off here.
        return super().validation_step(batch)

    # ── MC inference ─────────────────────────────────────────────────────

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_passes: int = None,
    ) -> dict:
        """
        Run MC Dropout inference on a pre-processed input tensor.

        Args:
            x:        FloatTensor of shape (B, C, [D,] H, W) on self.device
            n_passes: override self.n_mc_passes if desired

        Returns a dict with keys:
            'segmentation'       – argmax of mean_prob  (B, [D,] H, W)
            'mean_probability'   – mean softmax probs   (B, C, [D,] H, W)
            'variance'           – per-voxel variance   (B, C, [D,] H, W)
            'predictive_entropy' – predictive entropy   (B, [D,] H, W)
            'mutual_information' – epistemic uncertainty(B, [D,] H, W)
        """
        if n_passes is None:
            n_passes = self.n_mc_passes

        x = x.to(self.device)

        self.network.eval()
        enable_mc_dropout(self.network)

        samples = []
        with torch.no_grad():
            with autocast(self.device.type, enabled=True):
                for _ in range(n_passes):
                    logits = self.network(x)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    probs = torch.softmax(logits, dim=1)
                    samples.append(probs.cpu().float())

        # ── aggregate ────────────────────────────────────────────────────
        samples = torch.stack(samples, dim=0)   # (N, B, C, ...)

        mean_prob = samples.mean(dim=0)          # (B, C, ...)
        variance  = samples.var(dim=0)           # (B, C, ...)

        eps = 1e-8

        # Predictive entropy  H[y | x]
        pred_entropy = -(
            mean_prob * torch.log(mean_prob + eps)
        ).sum(dim=1)                             # (B, ...)

        # Expected entropy per pass  E_w[ H[y | x, w] ]
        entropy_per_pass = -(
            samples * torch.log(samples + eps)
        ).sum(dim=2)                             # (N, B, ...)
        expected_entropy = entropy_per_pass.mean(dim=0)  # (B, ...)

        # Mutual information  (epistemic uncertainty)
        mutual_info = pred_entropy - expected_entropy    # (B, ...)

        segmentation = mean_prob.argmax(dim=1)           # (B, ...)

        return {
            "segmentation":        segmentation,
            "mean_probability":    mean_prob,
            "variance":            variance,
            "predictive_entropy":  pred_entropy,
            "mutual_information":  mutual_info,
        }

    # ── convenience: save uncertainty maps ───────────────────────────────

    def save_uncertainty_as_nifti(
        self,
        uncertainty_dict: dict,
        output_dir: str,
        case_id: str,
        spacing: Union[Tuple, List] = None,
    ) -> None:
        """
        Save the outputs of predict_with_uncertainty() as NIfTI files.

        Requires nibabel (`pip install nibabel`).

        Args:
            uncertainty_dict: returned by predict_with_uncertainty()
            output_dir:       directory to write files into
            case_id:          filename prefix (e.g. 'case_0001')
            spacing:          voxel spacing (z, y, x). Uses identity if None.
        """

        os.makedirs(output_dir, exist_ok=True)

        if spacing is not None:
            affine = np.diag(list(spacing) + [1.0])
        else:
            affine = np.eye(4)

        keys_to_save = [
            "segmentation",
            "predictive_entropy",
            "mutual_information",
        ]

        for key in keys_to_save:
            data = uncertainty_dict[key]
            # Remove batch dim (assume B=1)
            arr = data[0].numpy().astype(np.float32)
            img = nib.Nifti1Image(arr, affine)
            out_path = os.path.join(output_dir, f"{case_id}_{key}.nii.gz")
            nib.save(img, out_path)
            self.print_to_log_file(f"Saved: {out_path}")

        # Save per-class variance volumes
        variance = uncertainty_dict["variance"][0]  # (C, ...)
        for c in range(variance.shape[0]):
            arr = variance[c].numpy().astype(np.float32)
            img = nib.Nifti1Image(arr, affine)
            out_path = os.path.join(
                output_dir, f"{case_id}_variance_class{c}.nii.gz"
            )
            nib.save(img, out_path)
            self.print_to_log_file(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# Subclass variants with different dropout rates
# — lets you pick from the CLI without changing code
# ─────────────────────────────────────────────

class nnUNetTrainerMCDropout05(nnUNetTrainerMCDropout):
    """dropout_p = 0.05"""
    dropout_p = 0.05


class nnUNetTrainerMCDropout20(nnUNetTrainerMCDropout):
    """dropout_p = 0.20"""
    dropout_p = 0.20


class nnUNetTrainerMCDropout50Passes(nnUNetTrainerMCDropout):
    """dropout_p = 0.1, n_mc_passes = 50"""
    dropout_p   = 0.1
    n_mc_passes = 50