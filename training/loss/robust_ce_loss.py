import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weight: torch.Tensor | None = None,   # shape [C], class-level weights (optional)
        ignore_index: int | None = None,
        reduction: str = "mean",                    # 'none' | 'mean' | 'sum'
        label_smoothing: float = 0.0,
        normalize_by_weight_sum: bool = True,       # normalize mean by sum of weights
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.normalize_by_weight_sum = normalize_by_weight_sum
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,          # [B, C, *] unnormalized scores
        target: torch.Tensor,          # [B, *] class indices (not one-hot)
        weight_map: torch.Tensor | None = None,  # [B, *] or broadcastable to target
    ) -> torch.Tensor:

        # elementwise CE: same spatial shape as target

        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weight,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # shape [B, *]

        # Build/align weights
        if weight_map is None:
            w = torch.ones_like(ce)
        else:
            # Allow broadcast; then match shape
            w = weight_map

        # Zero-out ignored locations in both loss and weights
        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index)
            ce = ce.masked_fill(ignore_mask, 0.0)
            w = w.masked_fill(ignore_mask, 0.0)

        # Apply pixel/instance weights
        w = w.clamp_min(0)

        loss = ce * w

        if self.reduction == "none":
            return loss

        if self.reduction == "sum":
            return loss.sum()

        #if self.reduction == "mean":
        #    return loss.mean()
        
        # weighted mean
        if self.normalize_by_weight_sum:
            denom = w.sum().clamp_min(1e-3 * w.numel())
            return loss.sum() / denom
        else:
            # unweighted mean over non-ignored elements
            count = (w != 0).sum().clamp_min(1)
            return loss.sum() / count
