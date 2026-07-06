import torch
import torch.nn as nn

from monai.losses import DiceLoss


class LossManager(nn.Module):

    def __init__(
        self,
        heatmap_weight=1.0,
        offset_weight=1.0,
        radius_weight=0.5,
        division_weight=1.0,
        embedding_weight=0.2,
        confidence_weight=0.25
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

    def heatmap_loss(
        self,
        prediction,
        target
    ):

        dice = self.dice(
            prediction,
            target
        )

        bce = self.bce(
            prediction,
            target
        )

        return dice + bce
    def confidence_loss(

    self,

    prediction,

    target

    ):

        return self.bce(

            prediction,

            target

     )
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