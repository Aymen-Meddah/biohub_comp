import tempfile
import unittest
from pathlib import Path


class ConfigTests(unittest.TestCase):

    def test_config_import_is_lazy(self):
        from config import Config

        self.assertGreaterEqual(Config.NUM_WORKERS, 0)
        self.assertEqual(Config.OUTPUT_DIR.name, "outputs")

    def test_dataset_validation_accepts_expected_structure(self):
        from config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train").mkdir()
            (root / "test").mkdir()

            original = Config.__class__._data_root
            Config.__class__._data_root = root
            try:
                self.assertTrue(Config.validate_dataset_exists(require_test=True))
            finally:
                Config.__class__._data_root = original


if __name__ == "__main__":
    unittest.main()
