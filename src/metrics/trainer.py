import torch 

class Trainer :
    def __init__(
            self,
            model ,
            criterion,
            optimizer,
            device):
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
    def train_one_epoch(
            self ,
            dataloder

    ):
        self.model.train()
        running_loss = 0.0
        for batch in dataloder:
            images = batch["image"].to(self.device)
            targets = images.clone()
            self.optimizer.zero_grad()
            prediction = self.model(images)

            losses = self.criterion(
                prediction,
                targets
            )
            loss = losses["loss"]
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        running_loss /= len(dataloder)
        return running_loss
