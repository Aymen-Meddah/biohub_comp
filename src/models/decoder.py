import torch
import torch.nn as nn

from src.models.blocks import ConvBlock
from src.models.blocks import ResidualBlock


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose3d(

            in_channels,

            out_channels,

            kernel_size=2,

            stride=2

        )

        self.conv = nn.Sequential(

            ConvBlock(

                out_channels + skip_channels,

                out_channels

            ),

            ResidualBlock(out_channels)

        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x


class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.up3 = UpBlock(
            128,
            64,
            64
        )

        self.up2 = UpBlock(
            64,
            32,
            32
        )

        self.up1 = UpBlock(
            32,
            16,
            16
        )

    def forward(
        self,
        features
    ):

        x = self.up3(

            features["bottleneck"],

            features["stage3"]

        )

        x = self.up2(

            x,

            features["stage2"]

        )

        x = self.up1(

            x,

            features["stage1"]

        )

        return x