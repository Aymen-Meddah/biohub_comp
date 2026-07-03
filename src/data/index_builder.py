from pathlib import Path

class IndexBuilder :
    def __init__(self ,train_diractory):
        self.train_diractory = Path(train_diractory)
        if not self.train_directory.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_directory}")
        self.samples = []

    def buiuld(self):
        self.samples.clear()
        zarr_files = sorted(self.train_directory.glob("*.zarr"))
        for zarr_path in zarr_files :
            dataset_name = zarr_path.stem
            geff_path = self.train_diractory / f"{dataset_name}.geff"

            if not geff_path.exists():
                continue

            self.samples.append({

                "dataset": dataset_name ,
                "zarr": zarr_path,
                "geff": geff_path
            })
        return self.samples
    def __getitem__(self, index):
        return len(self.samples)
    def __getitem__(self,index):
        return self.samples[index]
        
        