from torch.utils.data import DataLoader 
from src.data.dataset import BioHubDataset

def create_dataloader (
    data_directory,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    patch_size=(32, 96, 96),
    split="train"
):
    dataset = BioHubDataset(
        data_dir=data_directory,
        patch_size=patch_size,
        split=split
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return loader
