import torch
import torch.nn as nn 
from monai.networks.nets import DynUNet 

class Backbone(nn.Module):
    def __init__(self ,in_channels=1 ,featurs_channels=32):
        super().__init__()
        self.network =DynUNet (

            spatial_dims = 3 ,
            in_channels=in_channels,

            out_channels=feature_channels,

            kernel_size=[
                [3,3,3],
                [3,3,3],
                [3,3,3],
                [3,3,3],
                [3,3,3]
            ],

            strides=[
                [1,1,1],
                [2,2,2],
                [2,2,2],
                [2,2,2],
                [2,2,2]
            ],

            upsample_kernel_size=[
                [2,2,2],
                [2,2,2],
                [2,2,2],
                [2,2,2]
            ],

            res_block=True,

            deep_supervision=False

        )
        
    def forward(self , x):
        features = self.network(x)
        return features
    