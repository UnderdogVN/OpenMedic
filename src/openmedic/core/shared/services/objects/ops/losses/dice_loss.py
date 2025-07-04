import torch
import torch.nn.functional as F

import openmedic.core.shared.services.objects.loss_function as lf
import openmedic.core.shared.services.plans.registry as registry


class MultiClassDiceLoss(lf.OpenMedicLossOpBase):
    def __init__(self, n_classes: int, smooth=1.0, include_background: bool = False):
        super().__init__()
        self.n_classes: int = n_classes
        self.smooth: float = smooth
        self.include_background: int = include_background

    def forward(self, gts_pred: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        """
        gts_pred: [B, C, H, W] — raw model outputs (C = number of foreground classes)
        gts: [B, H, W] — class labels in [0, C-1], background/ignore = `ignore_index` (e.g., 255)
        """
        gts_pred_shapes: tuple = gts_pred.shape
        gts_shapes: tuple = gts.shape
        assert (
            len(gts_pred_shapes) == 4
        ), f"`gts_pred.shape` expect to 4 but return {len(gts_pred_shapes)}"
        assert (
            len(gts_shapes) == 3
        ), f"`gts_shapes.shape` expect to 3 but return {len(gts_shapes)}"

        # valid_mask: exclude ignore_index (e.g., 255)
        valid_mask: torch.Tensor
        if not self.include_background:
            valid_mask = gts != 0
        else:
            valid_mask = gts

        # Clone and safely replace invalid values for one-hot
        safe_targets: torch.Tensor = gts.clone()
        safe_targets[~valid_mask] = 0  # won't matter; masked out later

        # One-hot encode: [B, H, W] → [B, H, W, C] → [B, C, H, W]
        one_hot: torch.Tensor = (
            F.one_hot(safe_targets, num_classes=self.n_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Compute softmax probabilities
        probs: torch.Tensor = F.softmax(gts_pred, dim=1)

        # Apply mask
        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        # Compute Dice
        dims: tuple = (0, 2, 3)
        intersection: torch.Tensor = (probs * one_hot).sum(dim=dims)
        cardinality: torch.tensor = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice: torch.Tensor = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        return 1 - dice.mean()


def init():
    registry.LossRegister.register(loss_class=MultiClassDiceLoss)
