import tempfile
import unittest
from pathlib import Path

from src.data.index_builder import IndexBuilder


class IndexBuilderTests(unittest.TestCase):

    def test_build_finds_nested_zarr_and_geff_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_dir = root / "train"
            train_dir.mkdir()

            (train_dir / "sample.zarr").mkdir()
            (train_dir / "sample.geff").write_text("placeholder", encoding="utf-8")

            builder = IndexBuilder(root)
            samples = builder.build()

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["dataset"], "train_sample")
            self.assertEqual(samples[0]["zarr"], train_dir / "sample.zarr")
            self.assertEqual(samples[0]["geff"], train_dir / "sample.geff")


if __name__ == "__main__":
    unittest.main()
