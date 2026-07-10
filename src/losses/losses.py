import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss


class LossManager(nn.Module):
    """
    Loss manager for cell detection with CenterNet-style approach.
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
        focal_alpha=0.75,
        heatmap_pos_weight=2.0,
        confidence_pos_weight=2.0,
        use_dice=True,
        use_focal=True,
        use_bce=True,
        dice_weight=1.0,
        focal_weight=0.5,
        bce_weight=1.0,
        eps=1e-6,
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

        self.dice = DiceLoss(sigmoid=True, smooth_nr=eps, smooth_dr=eps)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def heatmap_loss(self, prediction, target):
        """Combined heatmap loss: Dice + Weighted BCE + Focal"""
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
            focal = self._focal_loss(prediction, target)
            loss = loss + self.focal_weight * focal

        return loss

    def confidence_loss(self, prediction, target):
        """Confidence loss for cell division detection"""
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
            focal = self._focal_loss(prediction, target)
            loss = loss + 0.5 * focal

        return loss

    def offset_loss(self, prediction, target):
        """L1 loss for offset regression - FIXED MASK"""
        # FIXED: Use proper mask for offset
        # Offset target has shape (B, 2, H, W) with small values
        # We mask where the heatmap is positive, not where offset is non-zero
        # But since we don't have heatmap here, we use a threshold
        
        # Method 1: Mask where offset magnitude is significant
        # offset values are typically 0 to 1 (fractional part)
        mask = (target.abs().sum(dim=1, keepdim=True) > 1e-3).float()
        
        if mask.sum() == 0:
            # If no valid pixels, return small loss to avoid NaN
            return torch.tensor(0.01, device=prediction.device)

        diff = (prediction - target).abs()
        return (diff * mask).sum() / (mask.sum() + self.eps)

    def radius_loss(self, prediction, target):
        """L1 loss for radius regression"""
        # Radius target: positive where cells exist
        mask = (target > 0).float()
        
        if mask.sum() == 0:
            return torch.tensor(0.01, device=prediction.device)

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

    def embedding_loss(self, prediction, target):
        """MSE loss for embeddings - FIXED MASK"""
        # FIXED: Use proper mask for embedding
        mask = (target.abs().sum(dim=1, keepdim=True) > 1e-3).float()
        
        if mask.sum() == 0:
            return torch.tensor(0.01, device=prediction.device)

        diff = (prediction - target) ** 2
        return (diff * mask).sum() / (mask.sum() + self.eps)

    def forward(self, outputs, targets):
        losses = {}

        losses["heatmap"] = self.heatmap_loss(
            outputs["heatmap"],
            targets["heatmap"]
        )

        losses["offset"] = self.offset_loss(
            outputs["offsets"],
            targets["offset"]
        )

        losses["radius"] = self.radius_loss(
            outputs["radius"],
            targets["radius"]
        )

        losses["division"] = self.division_loss(
            outputs["division"],
            targets["division"]
        )

        losses["embedding"] = self.embedding_loss(
            outputs["embedding"],
            targets["embedding"]
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

    def _focal_loss(self, prediction, target):
        """Fixed focal loss with correct alpha"""
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
        num_pos = target.sum().clamp(min=1.0)

        return focal_loss.sum() / num_pos