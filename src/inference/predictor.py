import torch

from src.data.zarr_reader import ZarrReader
from src.inference.decoder import CellDecoder


class Predictor:

    def __init__(self, model, device, threshold=0.30):
        self.model = model
        self.device = device
        self.decoder = CellDecoder(threshold=threshold)

    @torch.no_grad()
    def predict_zarr(self, zarr_path, dataset_name, array_key=None):
        reader = ZarrReader(zarr_path, array_key=array_key)
        predictions = []

        self.model.eval()
        for timepoint in range(reader.shape[0]):
            frame = reader.frame(timepoint)
            image = torch.as_tensor(
                frame,
                dtype=torch.float32,
                device=self.device
            )

            max_value = torch.max(image)
            if max_value > 0:
                image = image / max_value

            image = image.unsqueeze(0).unsqueeze(0)
            outputs = self.model(image)
            predictions.extend(
                self.decoder.decode(
                    outputs,
                    dataset_name,
                    timepoint
                )
            )

        return predictions
