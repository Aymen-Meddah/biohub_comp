import torch
import torch.nn as nn

from src.models.blocks import ConvBlock
from src.models.blocks import ResidualBlock


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(1, 32),
            ResidualBlock(32)
        )

        # Down 1
        self.down1 = ConvBlock(
            32,
            64,
            stride=2
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Down 2
        self.down2 = ConvBlock(
            64,
            128,
            stride=2
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Down 3
        self.down3 = ConvBlock(
            128,
            256,
            stride=2
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )

    def forward(self, x):

        s1 = self.stage1(x)

        s2 = self.stage2(self.down1(s1))

        s3 = self.stage3(self.down2(s2))

        b = self.bottleneck(self.down3(s3))

        return {
            "stage1": s1,
            "stage2": s2,
            "stage3": s3,
            "bottleneck": b
        }