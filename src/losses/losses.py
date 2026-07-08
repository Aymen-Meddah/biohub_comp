import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss


class LossManager(nn.Module):

    def __init__(
        self,
        heatmap_weight=1.0,
        offset_weight=1.0,
        radius_weight=0.5,
        division_weight=1.0,
        embedding_weight=0.2,
        confidence_weight=0.25,
        heatmap_pos_weight=2.0,
        confidence_pos_weight=2.0,
        focal_gamma=2.0,
        focal_alpha=0.25,
    ):

        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.radius_weight = radius_weight
        self.division_weight = division_weight
        self.embedding_weight = embedding_weight
        self.confidence_weight = confidence_weight
        self.dice = DiceLoss(sigmoid=True)

        self.bce = nn.BCEWithLogitsLoss()

        self.l1 = nn.L1Loss()

        self.mse = nn.MSELoss()

        self.heatmap_pos_weight = heatmap_pos_weight
        self.confidence_pos_weight = confidence_pos_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def heatmap_loss(
        self,
        prediction,
        target
    ):

        target = target.float()
        prediction = prediction.float()

        dice = self.dice(
            prediction,
            target
        )

        pos_weight = torch.as_tensor(
            self.heatmap_pos_weight,
            device=prediction.device,
        )
        bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="mean",
        )(prediction, target)

        focal = self._focal_loss(
            prediction,
            target,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
        )

        return dice + bce + 0.5 * focal
    def confidence_loss(

    self,

    prediction,

    target

    ):

        target = target.float()
        prediction = prediction.float()
        pos_weight = torch.as_tensor(
            self.confidence_pos_weight,
            device=prediction.device,
        )
        bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="mean",
        )(prediction, target)

        focal = self._focal_loss(
            prediction,
            target,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
        )

        return bce + 0.5 * focal
    def offset_loss(
        self,
        prediction,
        target
    ):

        return self.l1(
            prediction,
            target
        )

    def radius_loss(
        self,
        prediction,
        target
    ):

        return self.l1(
            prediction,
            target
        )

    def division_loss(
        self,
        prediction,
        target
    ):

        return self.bce(
            prediction,
            target
        )

    def embedding_loss(
        self,
        prediction,
        target
    ):

        return self.mse(
            prediction,
            target
        )

    def forward(
        self,
        outputs,
        targets
    ):

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

        losses["confidence"] = self.confidence_loss(

            outputs["confidence"],

            targets["confidence"]

        )
        total = (

            self.heatmap_weight * losses["heatmap"]

            +

            self.offset_weight * losses["offset"]

            +

            self.radius_weight * losses["radius"]

            +

            self.division_weight * losses["division"]

            +

            self.embedding_weight * losses["embedding"]

            +

            self.confidence_weight *

            losses["confidence"]
        )

        losses["total"] = total

        return losses

    def _focal_loss(self, prediction, target, gamma=2.0, alpha=0.25):
        if target.numel() == 0:
            return torch.tensor(0.0, device=prediction.device)

        prediction = prediction.float()
        target = target.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            prediction,
            target,
            reduction="none",
        )
        probas = torch.sigmoid(prediction)
        p_t = target * probas + (1 - target) * (1 - probas)
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        modulating_factor = (1 - p_t) ** gamma

        return (alpha_factor * modulating_factor * bce_loss).mean()
