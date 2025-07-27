import torch
import torch.nn.functional as F

import openmedic.core.shared.services.objects.loss_function as lf
import openmedic.core.shared.services.plans.registry as registry


class MultiClassDiceLoss(lf.OpenMedicLossOpBase):
    def __init__(self, n_classes: int, smooth=1.0, include_background: bool = False):
        super().__init__()
        self.n_classes: int = n_classes
        self.smooth: float = smooth
        self.include_background: bool = include_background  # Fixed: should be bool

    def forward(self, gts_pred: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        """
        gts_pred: [B, C, H, W] — raw model outputs (C = number of classes)
        gts: [B, H, W] — class labels in [0, C-1]
        """
        gts_pred_shapes: tuple = gts_pred.shape
        gts_shapes: tuple = gts.shape
        assert (
            len(gts_pred_shapes) == 4
        ), f"`gts_pred.shape` expect to 4 but return {len(gts_pred_shapes)}"
        assert (
            len(gts_shapes) == 3
        ), f"`gts_shapes.shape` expect to 3 but return {len(gts_shapes)}"

        # Ensure ground truth values are within valid range
        assert gts.min() >= 0 and gts.max() < self.n_classes, f"Ground truth values should be in [0, {self.n_classes-1}]"

        # One-hot encode ground truth: [B, H, W] → [B, C, H, W]
        one_hot: torch.Tensor = F.one_hot(gts, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        # Compute softmax probabilities from model output (differentiable)
        probs: torch.Tensor = F.softmax(gts_pred, dim=1)

        # Determine which classes to include in loss computation
        if not self.include_background:
            # Exclude background class (class 0) from loss computation
            class_indices = list(range(1, self.n_classes))  # Skip background class
            one_hot = one_hot[:, class_indices, :, :]
            probs = probs[:, class_indices, :, :]
        else:
            # Include all classes including background
            class_indices = list(range(self.n_classes))

        # Compute Dice coefficient for each class (differentiable)
        dims: tuple = (0, 2, 3)  # Sum over batch, height, width dimensions
        
        # Intersection: sum of element-wise product
        intersection: torch.Tensor = (probs * one_hot).sum(dim=dims)
        
        # Cardinality: sum of probabilities + sum of ground truth
        cardinality: torch.Tensor = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        
        # Dice coefficient with smoothing
        dice: torch.Tensor = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Return 1 - dice.mean() (lower is better for loss)
        return 1 - dice.mean()


def init():
    registry.LossRegister.register(loss_class=MultiClassDiceLoss)
