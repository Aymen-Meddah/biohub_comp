import numpy as np

from scipy.ndimage import maximum_filter


class PeakFinder:

    def __init__(

        self,

        threshold=0.30,

        window_size=3

    ):

        self.threshold = threshold

        self.window_size = window_size

    def find(

        self,

        heatmap

    ):

        heatmap = np.asarray(
            heatmap,
            dtype=np.float32
        )

        local_max = maximum_filter(

            heatmap,

            size=self.window_size,

            mode="nearest"

        )

        peaks = (

            heatmap == local_max

        ) & (

            heatmap >= self.threshold

        )

        coordinates = np.argwhere(
            peaks
        )

        detections = []

        for z, y, x in coordinates:

            detections.append(

                {

                    "z": int(z),

                    "y": int(y),

                    "x": int(x),

                    "score": float(

                        heatmap[
                            z,
                            y,
                            x
                        ]

                    )

                }

            )

        return detections