import torch

from tqdm import tqdm

from src.validation.matcher import Matcher
from src.validation.metrics import DetectionMetrics
from src.inference.decoder import CellDecoder


class Evaluator:

    def __init__(

        self,

        model,

        device,

        threshold=0.30,

        max_distance=10.0

    ):

        self.model = model

        self.device = device

        self.decoder = CellDecoder(

            threshold=threshold

        )

        self.matcher = Matcher(

            max_distance=max_distance

        )

        self.metrics = DetectionMetrics()

    @torch.no_grad()

    def evaluate(

        self,

        dataloader

    ):

        self.model.eval()

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for batch in tqdm(dataloader):

            images = batch["image"].to(

                self.device

            )

            outputs = self.model(

                images

            )

            predictions = self.decoder.decode(

                outputs,

                batch["dataset"][0],

                int(batch["timepoint"][0])

            )

            targets = batch["cells"][0]

            matches, fp, fn = self.matcher.match(

                predictions,

                targets

            )

            total_tp += len(matches)
            total_fp += len(fp)
            total_fn += len(fn)

        if len(dataloader) == 0:
            return {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)

        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
