from typing import Any

from torch.utils.data import DataLoader

from src.data.dataset import BioHubDataset


def create_dataloader(
    data_directory: str,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    patch_size: tuple[int, int, int] = (32, 96, 96),
    split: str = "train",
) -> DataLoader[Any]:
    dataset = BioHubDataset(
        data_dir=data_directory,
        patch_size=patch_size,
        split=split,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and __import__("torch").cuda.is_available(),
        drop_last=False,
    )

    return loader
