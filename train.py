import torch
from torch.utils.data import DataLoader

from config import Config
from src.data.collate import collate_fn
from src.data.dataset import CellTrackingDataset
from src.losses.losses import LossManager
from src.models.model import BioHubModel
from src.tracking.trainer import Trainer
from src.validation.validator import Validator


Config.create_directories()


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": Config.PIN_MEMORY,
        "collate_fn": collate_fn,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR

    return DataLoader(**kwargs)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = CellTrackingDataset(
        data_dir=Config.TRAIN_DIR,
        split="train",
        patch_size=Config.IMAGE_SIZE,
        sigma=Config.HEATMAP_SIGMA,
        embedding_dim=Config.EMBEDDING_DIM,
    )

    validation_dataset = CellTrackingDataset(
        data_dir=Config.TRAIN_DIR,
        split="validation",
        patch_size=Config.IMAGE_SIZE,
        sigma=Config.HEATMAP_SIGMA,
        embedding_dim=Config.EMBEDDING_DIM,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=Config.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.VALIDATION_NUM_WORKERS,
    )

    model = BioHubModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.SCHEDULER_T_MAX,
    )

    criterion = LossManager()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        use_amp=Config.AMP and device.type == "cuda",
    )

    validator = Validator(
        model=model,
        device=device,
        threshold=Config.CONFIDENCE_THRESHOLD,
        max_distance=Config.MATCHING_DISTANCE,
    )

    best_score = 0.0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        train_history = trainer.train_epoch(train_loader)
        validation = validator.validate(validation_loader)
        score = validation["metrics"]["f1"]

        print(train_history)
        print(validation["metrics"])

        if score > best_score:
            best_score = score

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "score": score,
                },
                Config.CHECKPOINT_PATH,
            )
            print("Best model saved.")


if __name__ == "__main__":
    main()
