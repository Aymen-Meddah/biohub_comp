import numpy as np

class HeatmapBuilder :
    def __init__(self ,sigma=2.5):
        self.sigma = sigma
    def generate(self , patch_shape ,cells):
        heatmap = np.zeros(patch_shape,dtype = np.float32)
        zz ,yy,xx = np.indices(
            patch_shape
        )
        for cell in cells:

            z = cell["z"]
            y = cell["y"]
            x = cell["x"]

            gaussian = np.exp(

                -(
                    (zz-z)**2 +
                    (yy-y)**2 +
                    (xx-x)**2
                )

                /

                (2*self.sigma*self.sigma)

            )

            heatmap = np.maximum(
                heatmap,
                gaussian
            )

        return heatmap