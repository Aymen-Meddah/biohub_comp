import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from src.data.collate import collate_fn
from src.data.dataset import CellTrackingDataset
from src.losses.losses import LossManager
from src.models.model import BioHubModel
from src.tracking.trainer import Trainer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import Logger
from src.utils.report import generate_training_report
from src.validation.validator import Validator

Config.create_directories()

LOGGER = Logger(str(Config.LOG_DIR / "training.log"))
CHECKPOINT_MANAGER = CheckpointManager(str(Config.CHECKPOINT_DIR))


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": Config.PIN_MEMORY and torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR

    return DataLoader(**kwargs)


def save_summary(history, report_path):
    report_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def main():
    Config.validate_dataset_exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        num_workers=0,
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=Config.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = BioHubModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.LR_SCHEDULER_MODE,
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=Config.LR_SCHEDULER_MIN_LR,
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

    start_epoch = 0
    best_score = 0.0
    checkpoint_path = Config.CHECKPOINT_DIR / "latest_checkpoint.pth"

    if checkpoint_path.exists():
        checkpoint = CHECKPOINT_MANAGER.load(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            filename=checkpoint_path.name,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("score", 0.0))
        LOGGER.info(f"Resumed from epoch {start_epoch}")

    history = []
    epochs_since_improvement = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        LOGGER.info(f"Epoch {epoch + 1}/{Config.EPOCHS}")

        train_history = trainer.train_epoch(train_loader)
        validation = validator.validate(validation_loader)
        score = float(validation["metrics"]["f1"])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        improved = score > best_score + Config.EARLY_STOPPING_MIN_DELTA
        if improved:
            best_score = score
            epochs_since_improvement = 0
            CHECKPOINT_MANAGER.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                score=score,
                config={"epochs": Config.EPOCHS, "batch_size": Config.BATCH_SIZE},
                filename="best_model.pth",
            )
            LOGGER.info("Best checkpoint saved.")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= Config.EARLY_STOPPING_PATIENCE:
            LOGGER.warning(
                "Early stopping triggered: validation F1 did not improve for "
                f"{Config.EARLY_STOPPING_PATIENCE} epochs."
            )
            CHECKPOINT_MANAGER.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                score=best_score,
                config={"epochs": Config.EPOCHS, "batch_size": Config.BATCH_SIZE},
                filename="latest_checkpoint.pth",
            )
            save_summary(history, Config.OUTPUT_DIR / "training_summary.json")
            break

        entry = {
            "epoch": epoch + 1,
            "train": train_history,
            "validation": validation["metrics"],
            "best_score": best_score,
        }
        history.append(entry)

        LOGGER.info(json.dumps(entry, indent=2))

        CHECKPOINT_MANAGER.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            score=best_score,
            config={"epochs": Config.EPOCHS, "batch_size": Config.BATCH_SIZE},
            filename="latest_checkpoint.pth",
        )

        save_summary(history, Config.OUTPUT_DIR / "training_summary.json")

    LOGGER.info("Training completed.")
    generate_training_report(
        Config.OUTPUT_DIR / "training_summary.json",
        Config.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
