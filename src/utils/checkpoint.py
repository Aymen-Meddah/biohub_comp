from pathlib import Path

import torch


class CheckpointManager:

    def __init__(

        self,

        checkpoint_dir

    ):

        self.checkpoint_dir = Path(

            checkpoint_dir

        )

        self.checkpoint_dir.mkdir(

            parents=True,

            exist_ok=True

        )

    def save(

        self,

        model,

        optimizer,

        scheduler,

        epoch,

        score,

        config,

        filename="checkpoint.pth"

    ):

        checkpoint = {

            "epoch": epoch,

            "score": score,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict()

            if scheduler is not None

            else None,

            "config": config

        }

        path = self.checkpoint_dir / filename

        torch.save(

            checkpoint,

            path

        )

        return path

    def load(

        self,

        model,

        optimizer=None,

        scheduler=None,

        filename="checkpoint.pth",

        device="cpu"

    ):

        path = self.checkpoint_dir / filename

        checkpoint = torch.load(

            path,

            map_location=device

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        if optimizer is not None:

            optimizer.load_state_dict(

                checkpoint["optimizer_state_dict"]

            )

        if (

            scheduler is not None

            and

            checkpoint["scheduler_state_dict"] is not None

        ):

            scheduler.load_state_dict(

                checkpoint["scheduler_state_dict"]

            )

        return {

            "epoch": checkpoint["epoch"],

            "score": checkpoint["score"],

            "config": checkpoint["config"]

        }

    def exists(

        self,

        filename="checkpoint.pth"

    ):

        return (

            self.checkpoint_dir /

            filename

        ).exists()