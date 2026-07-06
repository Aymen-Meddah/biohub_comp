import numpy as np


class DivisionBuilder:

    def __init__(self):
        pass

    def generate(
        self,
        patch_shape,
        cells
    ):

        division = np.zeros(
            patch_shape,
            dtype=np.float32
        )

        for cell in cells:

            if not cell.get("division", False):
                continue

            z = int(round(cell["z"]))
            y = int(round(cell["y"]))
            x = int(round(cell["x"]))

            if (
                0 <= z < patch_shape[0]
                and
                0 <= y < patch_shape[1]
                and
                0 <= x < patch_shape[2]
            ):

                division[z, y, x] = 1.0

        return division