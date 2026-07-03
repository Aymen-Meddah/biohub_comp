from pathlib import Path
import zarr 
import numpy as np 

class ZarrReader :
    def __init__(self , zarr_path):
        self.zarr_path = Path(zarr_path)

        if not self.zarr_path.exists():
            raise FileNotFoundError(
                f"zarr file not found : {self.zarr_path}"
            )
        self.volume = zarr.open(
            self.zarr_path,
            mode="r"

        )["0"]

    @property
    def dtype(self):
        return self.volume.dtype
    def frame(self ,t):
        return self.volume[t]
    def voxel(self , t ,z, y, x):
        return self.volume[t ,z ,y ,x]
    
    def patch(
            self ,
            t,
            z0 ,
            z1 ,
            y0 ,
            y1 ,
            x0 ,
            x1 
    ):
        return self.volume[t ,z0 :z1 ,y0 :y1 ,x0 :x1 ]
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