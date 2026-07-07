from pathlib import Path


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
        for root in search_roots:
            if not root.exists():
                continue

            for zarr_path in sorted(root.glob("**/*.zarr")):
                if zarr_path in seen:
                    continue
                seen.add(zarr_path)

                dataset_name = zarr_path.stem
                geff_path = zarr_path.with_suffix(".geff")

                if not geff_path.exists():
                    continue

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
