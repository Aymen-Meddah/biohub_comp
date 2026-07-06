import torch 
import torch.nn as nn 
from src.models.dynunet import BioHubDynUNet

class MultiTaskDynUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BioHubDynUNet(
            in_channels=1,
            out_channels=32
        )
        self.heatmap_head = nn.Conv3d(
            32,
            1,
            kernel_size=1
        )
        self.offset_head = nn.Conv3d(
            32,
            3,
            kernel_size=1
        )
    def forward(self, x):

        features = self.backbone(x)

        heatmap = self.heatmap_head(features)

        offsets = self.offset_head(features)

        division = self.division_head(features)

        return {

            "heatmap": heatmap,

            "offsets": offsets,

            "division": division

        }    