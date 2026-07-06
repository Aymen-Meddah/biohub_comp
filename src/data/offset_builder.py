import numpy as np


class OffsetBuilder:

    def __init__(self):
        pass

    def generate(
        self,
        patch_shape,
        cells
    ):
        """
        Parameters
        ----------
        patch_shape : tuple
            (Z, Y, X)

        cells : list
            Local cell coordinates inside the patch.

        Returns
        -------
        offset_map : np.ndarray
            Shape = (3, Z, Y, X)
            Channel 0 -> Δz
            Channel 1 -> Δy
            Channel 2 -> Δx
        """

        offset = np.zeros(
            (3,) + patch_shape,
            dtype=np.float32
        )

        for cell in cells:

            z = float(cell["z"])
            y = float(cell["y"])
            x = float(cell["x"])

            zi = int(round(z))
            yi = int(round(y))
            xi = int(round(x))

            if (
                0 <= zi < patch_shape[0]
                and 0 <= yi < patch_shape[1]
                and 0 <= xi < patch_shape[2]
            ):

                offset[0, zi, yi, xi] = z - zi
                offset[1, zi, yi, xi] = y - yi
                offset[2, zi, yi, xi] = x - xi

        return offset