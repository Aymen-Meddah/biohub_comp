import torch 
import numpy as np

from scipy.ndimage import maximum_filter

class CellDetector :
    def __init__(
            self,
            threshold=0.5,
            footprint=3
    ):
        self.threshold = threshold
        self.footprint = footprint

    def detect(self , heatmap):

        if torch.is_tensor(heatmap):
            heatmap = heatmap.detach().cpu().numpy()
            maxima = maximum_filter(
                heatmap,
                size=self.footprint
            )
        peaks = (heatmap == maxima)
        peaks &= (heatmap >self.threshold)
        coords = np.argwhere(peaks)
        cells = []

        for z,y,x in coords :
            cells.append({

                "z": int(z),

                "y": int(y),

                "x": int(x),

                "score": float(
                    heatmap[z, y, x]
                )

            })
            return cells