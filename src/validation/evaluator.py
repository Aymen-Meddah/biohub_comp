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

        total = {

            "tp":0,

            "fp":0,

            "fn":0,

            "precision":0.0,

            "recall":0.0,

            "f1":0.0

        }

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

            result = self.metrics.compute(

                matches,

                fp,

                fn

            )

            for key in total:

                total[key] += result[key]

        n = len(dataloader)

        if n == 0:
            return {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        for key in [

            "precision",

            "recall",

            "f1"

        ]:

            total[key] /= n

        return total
