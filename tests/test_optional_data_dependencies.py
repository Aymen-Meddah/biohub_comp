import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


class OptionalDataDependencyTests(unittest.TestCase):

    @unittest.skipIf(
        importlib.util.find_spec("zarr") is None,
        "zarr is not installed"
    )
    def test_zarr_reader_reads_patch(self):
        import zarr

        from src.data.zarr_reader import ZarrReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.zarr"
            root = zarr.open(path, mode="w")
            root.create_array(
                "0",
                data=np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
            )

            reader = ZarrReader(path)
            patch = reader.patch(
                t=0,
                z0=0,
                z1=2,
                y0=0,
                y1=3,
                x0=0,
                x1=4,
            )

            self.assertEqual(patch.shape, (2, 3, 4))

    @unittest.skipIf(
        importlib.util.find_spec("geff") is None,
        "geff is not installed"
    )
    def test_geff_reader_dependency_available(self):
        from src.data.geff_reader import GEFFReader, GeffReader

        self.assertIs(GEFFReader, GeffReader)


if __name__ == "__main__":
    unittest.main()
