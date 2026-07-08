from pathlib import Path
from typing import Any, Optional

import numpy as np


class ZarrReader:

    def __init__(self, zarr_path: str | Path, array_key: Optional[str] = None):
        self.zarr_path = self._normalize_path(zarr_path)
        self._validate_path_exists(self.zarr_path)

        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "The 'zarr' package is required to read BioHub zarr files. "
                "Install the project requirements before loading data."
            ) from exc

        try:
            store = zarr.open(self.zarr_path, mode="r")
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Unable to open Zarr data from path: {self.zarr_path}. "
                "Verify that the path is a valid Zarr archive or group."
            ) from exc

        self.volume = self._select_array(store, zarr, array_key)

    @staticmethod
    def _validate_path_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Zarr file not found: {path}")

        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(
                f"Zarr path exists but is not a file or directory: {path}"
            )

    @staticmethod
    def _normalize_path(zarr_path: Optional[str | Path]) -> Path:
        if zarr_path is None:
            raise ValueError(
                "Zarr path is empty. Check that the dataset sample points to a valid .zarr file."
            )

        if isinstance(zarr_path, Path):
            candidate = zarr_path
        else:
            candidate = str(zarr_path).strip()

        if not candidate:
            raise ValueError(
                "Zarr path is empty. Check that the dataset sample points to a valid .zarr file."
            )

        path = Path(candidate).expanduser()
        if path == Path(".") or str(path).strip() in {"", "."}:
            raise ValueError(
                "Zarr path resolves to the current working directory. Provide a real .zarr path."
            )

        return path

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
