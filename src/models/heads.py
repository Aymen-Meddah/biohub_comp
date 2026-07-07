import torch
import torch.nn as nn


class BaseHead(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=64
    ):

        super().__init__()

        self.head = nn.Sequential(

            nn.Conv3d(
                in_channels,
                max(8, hidden_channels // 2),
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(max(8, hidden_channels // 2)),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                max(8, hidden_channels // 2),
                max(8, hidden_channels // 2),
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(max(8, hidden_channels // 2)),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                max(8, hidden_channels // 2),
                out_channels,
                kernel_size=1
            )

        )

    def forward(self, x):

        return self.head(x)


class HeatmapHead(BaseHead):

    def __init__(self, in_channels):

        super().__init__(
            in_channels,
            1
        )


class OffsetHead(BaseHead):

    def __init__(self, in_channels):

        super().__init__(
            in_channels,
            3
        )


class RadiusHead(BaseHead):

    def __init__(self, in_channels):

        super().__init__(
            in_channels,
            1
        )


class DivisionHead(BaseHead):

    def __init__(self, in_channels):

        super().__init__(
            in_channels,
            1
        )


class EmbeddingHead(BaseHead):

    def __init__(
        self,
        in_channels,
        embedding_dim=16
    ):

        super().__init__(
            in_channels,
            embedding_dim
        )
class ConfidenceHead(BaseHead):

    def __init__(self, in_channels):

        super().__init__(
            in_channels,
            1
        )