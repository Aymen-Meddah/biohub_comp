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
        zarr_files = sorted(self.data_directory.glob("*.zarr"))

        for zarr_path in zarr_files:
            dataset_name = zarr_path.stem
            geff_path = self.data_directory / f"{dataset_name}.geff"

            if not geff_path.exists():
                continue

            self.samples.append({

                "dataset": dataset_name,
                "zarr": zarr_path,
                "geff": geff_path,
            })
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
