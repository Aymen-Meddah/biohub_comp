import torch

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from tqdm import tqdm


class Trainer:

    def __init__(

        self,

        model,

        optimizer,

        criterion,

        scheduler,

        device,

        use_amp=True,

        gradient_clip=5.0

    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.scheduler = scheduler

        self.device = device

        self.use_amp = use_amp

        self.gradient_clip = gradient_clip

        self.scaler = GradScaler(
            enabled=use_amp
        )

    def train_epoch(

        self,

        loader

    ):

        self.model.train()

        history = {

            "heatmap":0.0,

            "offset":0.0,

            "radius":0.0,

            "division":0.0,

            "embedding":0.0,

            "total":0.0

        }

        progress = tqdm(loader)

        for batch in progress:

            image = batch["image"].to(
                self.device
            )

            targets = {

                "heatmap":batch["heatmap"].to(self.device),

                "offset":batch["offset"].to(self.device),

                "radius":batch["radius"].to(self.device),

                "division":batch["division"].to(self.device),

                "embedding":batch["embedding"].to(self.device)

            }

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(

                enabled=self.use_amp

            ):

                outputs = self.model(

                    image

                )

                losses = self.criterion(

                    outputs,

                    targets

                )

            self.scaler.scale(

                losses["total"]

            ).backward()

            self.scaler.unscale_(

                self.optimizer

            )

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.gradient_clip

            )

            self.scaler.step(

                self.optimizer

            )

            self.scaler.update()

            if self.scheduler is not None:

                self.scheduler.step()

            for k in history:

                history[k] += losses[k].item()

            progress.set_postfix(

                loss=losses["total"].item()

            )

        for k in history:

            history[k] /= len(loader)

        return history