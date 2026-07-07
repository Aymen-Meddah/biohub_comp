from pathlib import Path
from typing import Any, Optional

import numpy as np


class ZarrReader:

    def __init__(self, zarr_path: str | Path, array_key: Optional[str] = None):
        self.zarr_path = Path(zarr_path)

        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "The 'zarr' package is required to read BioHub zarr files. "
                "Install the project requirements before loading data."
            ) from exc

        store = zarr.open(self.zarr_path, mode="r")
        self.volume = self._select_array(store, zarr, array_key)

    @staticmethod
    def _select_array(store: Any, zarr_module: Any, array_key: Optional[str] = None):
        if hasattr(store, "shape") and hasattr(store, "dtype"):
            return store

        if array_key is not None:
            return store[array_key]

        preferred_keys = ("0", "raw", "image", "images", "volume")
        for key in preferred_keys:
            if key in store and hasattr(store[key], "shape") and hasattr(store[key], "dtype"):
                return store[key]

        arrays = [
            key
            for key, value in store.items()
            if hasattr(value, "shape") and hasattr(value, "dtype")
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
