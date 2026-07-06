import torch

from torch.utils.data import DataLoader

from src.data.dataset import CellTrackingDataset

from src.models.model import CellTrackingModel

from src.losses.losses import MultiTaskLoss

from src.training.trainer import Trainer

from src.validation.validator import Validator

from config import Config

def main():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )

    train_dataset = CellTrackingDataset(

        split="train"

    )

    validation_dataset = CellTrackingDataset(

        split="validation"

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=Config.BATCH_SIZE,

        shuffle=True,

        num_workers=Config.NUM_WORKERS,

        pin_memory=Config.PIN_MEMORY

    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=Config.VALIDATION_BATCH_SIZE,

        shuffle=False,

        num_workers=Config.VALIDATION_NUM_WORKERS,

        pin_memory=Config.PIN_MEMORY

    )

    model = CellTrackingModel().to(

        device

    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=Config.LEARNING_RATE,

        weight_decay=Config.WEIGHT_DECAY

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=Config.SCHEDULER_T_MAX

    )

    criterion = MultiTaskLoss()

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        criterion=criterion,

        scheduler=scheduler,

        device=device

    )

    validator = Validator(

        model=model,

        device=device

    )

    epochs = Config.EPOCHS
    best_score = 0.0
    Config.CHECKPOINT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    for epoch in range(epochs):

        print(

            f"\nEpoch {epoch+1}/{epochs}"

        )

        train_history = trainer.train_epoch(

            train_loader

        )

        validation = validator.validate(

            validation_loader

        )

        score = validation["metrics"]["f1"]

        print(

            train_history

        )

        print(

            validation["metrics"]

        )

        if score > best_score:

            best_score = score

            torch.save(

                {

                    "epoch": epoch,

                    "model": model.state_dict(),

                    "optimizer": optimizer.state_dict(),

                    "score": score

                },

                Config.CHECKPOINT_PATH

            )

            print(

                "Best model saved."

            )


if __name__ == "__main__":

    main()
