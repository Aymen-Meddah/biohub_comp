import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):

    def __init__(

        self,

        channels

    ):

        super().__init__()

        self.lateral = nn.ModuleList(

            [

                nn.Conv3d(

                    c,

                    64,

                    kernel_size=1

                )

                for c in channels

            ]

        )

        self.output = nn.ModuleList(

            [

                nn.Sequential(

                    nn.Conv3d(

                        64,

                        64,

                        kernel_size=3,

                        padding=1,

                        bias=False

                    ),

                    nn.BatchNorm3d(64),

                    nn.ReLU(inplace=True)

                )

                for _ in channels

            ]

        )

    def forward(

        self,

        features

    ):

        pyramid = []

        last = None

        for i in reversed(

            range(

                len(features)

            )

        ):

            current = self.lateral[i](

                features[i]

            )

            if last is not None:

                last = F.interpolate(

                    last,

                    size=current.shape[-3:],

                    mode="trilinear",

                    align_corners=False

                )

                current = current + last

            current = self.output[i](

                current

            )

            pyramid.insert(

                0,

                current

            )

            last = current

        return pyramid