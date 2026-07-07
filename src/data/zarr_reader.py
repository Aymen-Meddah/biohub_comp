from pathlib import Path
import numpy as np
import zarr


class ZarrReader:

    def __init__(self, zarr_path, array_key=None):
        self.zarr_path = Path(zarr_path)

        if not self.zarr_path.exists():
            raise FileNotFoundError(
                f"Zarr file not found: {self.zarr_path}"
            )
        store = zarr.open(
            self.zarr_path,
            mode="r"
        )
        self.volume = self._select_array(store, array_key)

    @staticmethod
    def _select_array(store, array_key=None):
        if isinstance(store, zarr.Array):
            return store

        if array_key is not None:
            return store[array_key]

        preferred_keys = ("0", "raw", "image", "images", "volume")
        for key in preferred_keys:
            if key in store and isinstance(store[key], zarr.Array):
                return store[key]

        arrays = [
            key
            for key, value in store.items()
            if isinstance(value, zarr.Array)
        ]
        if not arrays:
            raise ValueError("No array found inside zarr group.")

        return store[sorted(arrays)[0]]

    @property
    def shape(self):
        return self.volume.shape

    @property
    def dtype(self):
        return self.volume.dtype

    def frame(self, t):
        return np.asarray(self.volume[t])

    def voxel(self, t, z, y, x):
        return self.volume[t, z, y, x]

    def patch(
        self,
        t,
        z0,
        z1,
        y0,
        y1,
        x0,
        x1
    ):
        return np.asarray(
            self.volume[t, z0:z1, y0:y1, x0:x1],
            dtype=np.float32
        )

    def statistics(self):
        return {
            "shape": self.shape,

            "dtype": str(self.dtype),

            "min": int(np.min(self.volume)),

            "max": int(np.max(self.volume)),
            "mean": float(np.mean(self.volume))
        }

    def __repr__(self):
        return (
            f"ZarrReader("
            f"shape={self.shape},"
            f"dtype={self.dtype})"
        )
