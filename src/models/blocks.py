import torch
import torch.nn as nn 

class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels ,kernel_size = 3 ,stride = 1 , padding = 1):
        super().__init__()
        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                bias = False
            ),
            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )
    def forward(self, x):

        return self.block(x)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.conv1 = ConvBlock(
            channels,
            channels
        )

        self.conv2 = nn.Sequential(

            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(channels)

        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out += identity

        out = self.relu(out)

        return out