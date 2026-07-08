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
            store = self._open_store(zarr, self.zarr_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Unable to open Zarr data from path: {self.zarr_path}. "
                "Verify that the path is a valid Zarr archive or group."
            ) from exc

        self.volume = self._select_array(store, zarr, array_key)

    @staticmethod
    def _open_store(zarr_module: Any, path: Path):
        path_str = str(path)

        if path.is_dir():
            for store_cls_name in ("DirectoryStore", "NestedDirectoryStore"):
                store_cls = getattr(zarr_module.storage, store_cls_name, None)
                if store_cls is None:
                    continue
                try:
                    return zarr_module.open(store_cls(path_str), mode="r")
                except Exception:
                    continue

            try:
                if hasattr(zarr_module, "open_consolidated"):
                    return zarr_module.open_consolidated(path_str)
            except Exception:
                pass

            try:
                return zarr_module.open(path_str, mode="r")
            except Exception:
                pass

        if path.is_file():
            if path.suffix == ".zarr":
                zip_store_cls = getattr(zarr_module, "ZipStore", None) or getattr(zarr_module.storage, "ZipStore", None)
                if zip_store_cls is not None:
                    try:
                        return zarr_module.open(zip_store_cls(path_str), mode="r")
                    except Exception:
                        pass

            try:
                return zarr_module.open(path_str, mode="r")
            except Exception:
                pass

            try:
                import fsspec
                fs = fsspec.filesystem("file")
                fs_store_cls = getattr(zarr_module.storage, "FSStore", None)
                if fs_store_cls is not None:
                    return zarr_module.open(fs_store_cls(path_str, mode="r", fs=fs), mode="r")
            except Exception:
                pass

        try:
            return zarr_module.open_group(path_str, mode="r")
        except Exception as exc:
            raise exc

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
