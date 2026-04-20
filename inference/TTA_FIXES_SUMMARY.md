# TTA Uncertainty Predictor - Fixes Applied

## Problem
The `tta_uncertainty_predictor.py` was experiencing freezing issues, likely due to improper device placement and checkpoint loading.

## Root Causes Identified

1. **Device Placement Issue**: Data loaded from preprocessing iterator was not being moved to the GPU/correct device before being passed to the network, causing type mismatches or slow CPU inference.

2. **Network Not in Eval Mode**: After loading checkpoints, the network was not explicitly set to `eval()` mode, which could cause issues with layers like BatchNorm and Dropout.

3. **Missing No-Grad Context**: Inference was not wrapped in `torch.no_grad()` context, consuming unnecessary GPU memory.

4. **Incomplete Initialization**: Network weights loaded but network may not have been properly transferred to device.

## Changes Made

### 1. Added `_finalize_network_initialization()` method
- Explicitly moves network to device
- Sets network to eval mode
- Validates that network was properly loaded
- Called immediately after `initialize_from_trained_model_folder()`

```python
def _finalize_network_initialization(self):
    if self.network is not None:
        self.network = self.network.to(self.device)
        self.network.eval()
        print(f'[TTA] Network moved to device {self.device} and set to eval mode')
    else:
        raise RuntimeError('[TTA] ERROR: Network is None. Did you call initialize_from_trained_model_folder?')
```

### 2. Enhanced `_predict_tta_all_probs()` method
- Explicitly moves input data to device at the start
- Wrapped entire inference loop in `torch.no_grad()` context
- Uses `.detach()` when converting to numpy to safely release gradients

### 3. Improved `predict_from_data_iterator()` data loading
- Ensures data from preprocessing iterator is properly converted to tensor
- Explicitly moves data to device before TTA processing
- Handles both numpy arrays and file paths robustly

### 4. Updated entry point
- Calls `_finalize_network_initialization()` after model loading
- Ensures network is ready before prediction starts

## Example Usage (Shell Script)

```bash
python /path/to/tta_uncertainty_predictor.py \
    --continue_prediction \
    -i "/path/to/input/images" \
    -o "/path/to/output/predictions" \
    -d 999 \
    -tr autoPET3_Trainer \
    -p nnUNetResEncUNetLPlansMultiTalent \
    -c 3d_fullres \
    -f 2 \
    -chk "checkpoint_best.pth"
```

## Output Structure

```
output_folder/
├── prediction_files/           # Averaged segmentation predictions
├── uncertainty_maps/           # Normalized uncertainty maps
│   ├── case_001_aleatoric_uncertainty.nii.gz
│   ├── case_001_epistemic_uncertainty.nii.gz
│   └── case_001_total_uncertainty.nii.gz
├── tta_predict_args.json       # TTA configuration used
├── dataset.json                # Dataset metadata
└── plans.json                  # nnU-Net plans
```

## Uncertainty Metrics

The predictor now correctly computes:

1. **Aleatoric Uncertainty** (Data uncertainty): Average entropy across TTA members
   - Represents variability due to input noise/ambiguity
   - Computed as: mean(H[p_i]) where H is Shannon entropy

2. **Epistemic Uncertainty** (Model uncertainty): Mutual information
   - Represents model's uncertainty about its own predictions
   - Computed as: H[mean(p_i)] - mean(H[p_i])

3. **Total Uncertainty**: Entropy of averaged predictions
   - Sum of aleatoric and epistemic uncertainty
   - Computed as: H[mean(p_i)]

All maps are normalized to [0, 1] range and saved as NIfTI files.

## Debugging Tips

If you still experience issues:

1. Check PyTorch GPU availability: `nvidia-smi` / `torch.cuda.is_available()`
2. Verify checkpoint file exists: `-chk checkpoint_best.pth`
3. Ensure input images have correct channel naming (_0000.nii.gz, etc.)
4. Check disk space for uncertainty map outputs
5. Monitor memory: Large 3D volumes with many TTA augmentations can use significant VRAM
6. If still freezing, run with `--verbose` flag for detailed logging
