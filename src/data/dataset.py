from pathlib import path
class BioHubDataset:
    """
    BioHub Dataset
    Version 1

    This version only indexes all datasets.
    No Zarr loading.
    No GEFF loading.
    """
    def __init__(self , train_dir):
        self.train_dir = path(train_dir)
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        self.samples = []
        self._build_index()

    def _build_index(self):
        zarr_files = sorted(self.train_dir.glob('*.zarr'))
        for zarr_path in zarr_files:
            dataset_name = zarr_path.stem
            geff_path = self.train_dir / f"{dataset_name}.geff"
            if geff_path.exists():
                self.samples.append({
                    'dataset_name': dataset_name,
                    'zarr_path': zarr_path,
                    'geff_path': geff_path
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
    