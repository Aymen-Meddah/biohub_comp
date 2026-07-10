import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss


class LossManager(nn.Module):
    """
    Loss manager for cell detection with CenterNet-style approach.

    CRITICAL FIXES from original:
    1. Focal loss alpha: 0.25 -> 0.75 (was INVERTED!)
    2. Focal loss normalization: by ALL pixels (not just positives)
    3. Mask uses adaptive threshold (handles small heatmap values)
    4. Gradient clipping to prevent explosion
    """

    def __init__(
        self,
        heatmap_weight=1.0,
        offset_weight=1.0,
        radius_weight=0.5,
        division_weight=1.0,
        embedding_weight=0.2,
        confidence_weight=0.25,
        focal_gamma=2.0,
        focal_alpha=0.75,  # FIXED: was 0.25 (inverted!)
        heatmap_pos_weight=2.0,
        confidence_pos_weight=2.0,
        use_dice=True,
        use_focal=True,
        use_bce=True,
        dice_weight=1.0,
        focal_weight=0.5,
        bce_weight=1.0,
        eps=1e-6,
        max_loss_value=100.0,
        mask_threshold=0.01,  # Low threshold for small heatmap values
    ):
        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.radius_weight = radius_weight
        self.division_weight = division_weight
        self.embedding_weight = embedding_weight
        self.confidence_weight = confidence_weight

        self.use_dice = use_dice
        self.use_focal = use_focal
        self.use_bce = use_bce
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.heatmap_pos_weight = heatmap_pos_weight
        self.confidence_pos_weight = confidence_pos_weight
        self.eps = eps
        self.max_loss_value = max_loss_value
        self.mask_threshold = mask_threshold

        self.dice = DiceLoss(sigmoid=True, smooth_nr=eps, smooth_dr=eps)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def heatmap_loss(self, prediction, target):
        """Combined heatmap loss with gradient clipping"""
        target = target.float()
        prediction = prediction.float()
        loss = 0.0

        if self.use_dice:
            dice = self.dice(prediction, target)
            loss = loss + self.dice_weight * dice

        if self.use_bce:
            pos_weight = torch.as_tensor(
                self.heatmap_pos_weight,
                device=prediction.device,
            )
            bce = F.binary_cross_entropy_with_logits(
                prediction, target,
                pos_weight=pos_weight,
                reduction="mean",
            )
            loss = loss + self.bce_weight * bce

        if self.use_focal:
            focal = self._focal_loss_fixed(prediction, target)
            loss = loss + self.focal_weight * focal

        return torch.clamp(loss, max=self.max_loss_value)

    def confidence_loss(self, prediction, target):
        """Confidence loss"""
        target = target.float()
        prediction = prediction.float()
        loss = 0.0

        if self.use_bce:
            pos_weight = torch.as_tensor(
                self.confidence_pos_weight,
                device=prediction.device,
            )
            bce = F.binary_cross_entropy_with_logits(
                prediction, target,
                pos_weight=pos_weight,
                reduction="mean",
            )
            loss = loss + bce

        if self.use_focal:
            focal = self._focal_loss_fixed(prediction, target)
            loss = loss + 0.5 * focal

        return torch.clamp(loss, max=self.max_loss_value)

    def offset_loss(self, prediction, target, heatmap_target=None):
        """
        L1 loss for offset regression - FIXED with adaptive mask

        Uses heatmap_target to create mask with adaptive threshold.
        """
        if heatmap_target is not None:
            hm_max = heatmap_target.max()
            if hm_max > 0.1:
                threshold = self.mask_threshold
            else:
                # For very small values, use 10% of max
                threshold = hm_max * 0.1

            mask = (heatmap_target > threshold).float()
        else:
            mask = (target.abs() > 1e-4).any(dim=1, keepdim=True).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device)

        diff = (prediction - target).abs()
        return (diff * mask).sum() / (mask.sum() * prediction.shape[1] + self.eps)

    def radius_loss(self, prediction, target, heatmap_target=None):
        """L1 loss for radius regression - FIXED"""
        if heatmap_target is not None:
            hm_max = heatmap_target.max()
            if hm_max > 0.1:
                threshold = self.mask_threshold
            else:
                threshold = hm_max * 0.1

            mask = (heatmap_target > threshold).float()
        else:
            mask = (target > 0).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device)

        diff = (prediction - target).abs()
        return (diff * mask).sum() / (mask.sum() + self.eps)

    def division_loss(self, prediction, target):
        """BCE loss for division classification"""
        target = target.float()
        prediction = prediction.float()

        num_pos = (target > 0.5).sum().float()
        num_neg = (target < 0.5).sum().float()

        if num_pos > 0:
            pos_weight = (num_neg / num_pos).clamp(min=1.0, max=10.0)
        else:
            pos_weight = torch.tensor(1.0, device=prediction.device)

        return F.binary_cross_entropy_with_logits(
            prediction, target,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def embedding_loss(self, prediction, target, heatmap_target=None):
        """MSE loss for embeddings - FIXED"""
        if heatmap_target is not None:
            hm_max = heatmap_target.max()
            if hm_max > 0.1:
                threshold = self.mask_threshold
            else:
                threshold = hm_max * 0.1

            mask = (heatmap_target > threshold).float()
        else:
            mask = (target.abs() > 1e-4).any(dim=1, keepdim=True).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device)

        diff = (prediction - target) ** 2
        return (diff * mask).sum() / (mask.sum() * prediction.shape[1] + self.eps)

    def forward(self, outputs, targets):
        losses = {}

        losses["heatmap"] = self.heatmap_loss(
            outputs["heatmap"],
            targets["heatmap"]
        )

        losses["offset"] = self.offset_loss(
            outputs["offsets"],
            targets["offset"],
            targets["heatmap"]  # Pass heatmap for masking
        )

        losses["radius"] = self.radius_loss(
            outputs["radius"],
            targets["radius"],
            targets["heatmap"]  # Pass heatmap for masking
        )

        losses["division"] = self.division_loss(
            outputs["division"],
            targets["division"]
        )

        losses["embedding"] = self.embedding_loss(
            outputs["embedding"],
            targets["embedding"],
            targets["heatmap"]  # Pass heatmap for masking
        )

        if "confidence" in outputs and "confidence" in targets:
            losses["confidence"] = self.confidence_loss(
                outputs["confidence"],
                targets["confidence"]
            )
        else:
            losses["confidence"] = torch.tensor(0.0, device=outputs["heatmap"].device)

        total = (
            self.heatmap_weight * losses["heatmap"]
            + self.offset_weight * losses["offset"]
            + self.radius_weight * losses["radius"]
            + self.division_weight * losses["division"]
            + self.embedding_weight * losses["embedding"]
            + self.confidence_weight * losses["confidence"]
        )

        losses["total"] = total
        return losses

    def _focal_loss_fixed(self, prediction, target):
        """
        FIXED focal loss - normalize by ALL pixels, not just positives.

        Original was dividing by num_pos only, causing explosion when
        num_pos is very small (1-2 pixels).
        """
        if target.numel() == 0:
            return torch.tensor(0.0, device=prediction.device)

        prediction = prediction.float()
        target = target.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            prediction, target, reduction="none"
        )

        probas = torch.sigmoid(prediction)
        p_t = target * probas + (1 - target) * (1 - probas)
        alpha_factor = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.focal_gamma)

        focal_loss = alpha_factor * modulating_factor * bce_loss

        # FIXED: Normalize by ALL pixels (not just positives!)
        return focal_loss.mean()