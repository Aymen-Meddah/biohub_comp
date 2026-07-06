import numpy as np


class RadiusBuilder:

    def __init__(self, default_radius=4.0):

        self.default_radius = default_radius

    def generate(
        self,
        patch_shape,
        cells
    ):

        radius = np.zeros(
            patch_shape,
            dtype=np.float32
        )

        for cell in cells:

            z = int(round(cell["z"]))
            y = int(round(cell["y"]))
            x = int(round(cell["x"]))

            if (
                0 <= z < patch_shape[0]
                and 0 <= y < patch_shape[1]
                and 0 <= x < patch_shape[2]
            ):

                if "radius" in cell:
                    radius[z, y, x] = float(cell["radius"])
                else:
                    radius[z, y, x] = self.default_radius

        return radius