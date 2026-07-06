import numpy as np


class EmbeddingBuilder:

    def __init__(
        self,
        embedding_dim=16
    ):

        self.embedding_dim = embedding_dim

    def generate(
        self,
        patch_shape,
        cells
    ):

        embedding = np.zeros(
            (self.embedding_dim,) + patch_shape,
            dtype=np.float32
        )

        for cell in cells:

            if "embedding" not in cell:
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

                vector = np.asarray(
                    cell["embedding"],
                    dtype=np.float32
                )

                embedding[:, z, y, x] = vector

        return embedding