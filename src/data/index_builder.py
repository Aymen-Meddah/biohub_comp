from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class IndexBuilder:

    def __init__(self, data_directory):
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_directory}"
            )
        self.samples = []

    def build(self):
        self.samples.clear()

        search_roots = []
        if self.data_directory.name in {"train", "test"}:
            search_roots.append(self.data_directory)
        else:
            if (self.data_directory / "train").exists():
                search_roots.append(self.data_directory / "train")
            if (self.data_directory / "test").exists():
                search_roots.append(self.data_directory / "test")
            if not search_roots:
                search_roots.append(self.data_directory)

        seen = set()
        zarr_candidates = []
        geff_candidates = []
        for root in search_roots:
            if not root.exists():
                continue

            for zarr_path in sorted(root.glob("**/*.zarr")):
                logger.info("Discovered zarr sample: %s", zarr_path)
                zarr_candidates.append(zarr_path)

            for geff_path in sorted(root.glob("**/*.geff")):
                logger.info("Discovered geff sample: %s", geff_path)
                geff_candidates.append(geff_path)

        for zarr_path in zarr_candidates:
            if zarr_path in seen:
                continue
            seen.add(zarr_path)

            dataset_name = zarr_path.stem
            geff_path = zarr_path.with_suffix(".geff")
            if geff_path.exists():
                logger.info("Accepted sample: %s paired with %s", zarr_path, geff_path)
            else:
                geff_path = self._find_geff_by_stem(zarr_path, geff_candidates)
                if geff_path is None:
                    logger.warning("Rejected sample %s: no matching .geff file found", zarr_path)
                    continue
                logger.info("Accepted sample: %s paired with %s", zarr_path, geff_path)

            try:
                relative_path = zarr_path.relative_to(self.data_directory)
            except ValueError:
                relative_path = zarr_path

            dataset_parts = list(relative_path.parts[:-1]) + [dataset_name]
            dataset_label = "_".join(part for part in dataset_parts if part)

            self.samples.append({
                "dataset": dataset_label or dataset_name,
                "zarr": zarr_path,
                "geff": geff_path,
            })

        return self.samples

    def _find_geff_by_stem(self, zarr_path, geff_candidates):
        stem = zarr_path.stem
        for geff_path in geff_candidates:
            if geff_path.stem == stem:
                return geff_path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
