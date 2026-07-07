import numpy as np
import torch
from src.tracking.node import Node
from src.inference.peak_finder import PeakFinder


class CellDecoder:

    def __init__(

        self,

        threshold=0.30

    ):

        self.peak_finder = PeakFinder(

            threshold=threshold

        )

    def decode(

    self,

    outputs,

    dataset,

    timepoint

    ):

        heatmap = torch.sigmoid(outputs["heatmap"][0, 0]).detach().cpu().numpy()

        offsets = outputs["offsets"][0].detach().cpu().numpy()

        radius = outputs["radius"][0, 0].detach().cpu().numpy()

        embedding = outputs["embedding"][0].detach().cpu().numpy()

        division = torch.sigmoid(outputs["division"][0, 0]).detach().cpu().numpy()

        confidence = torch.sigmoid(
            outputs["confidence"][0, 0]
        ).detach().cpu().numpy()

        peaks = self.peak_finder.find(heatmap)

        if peaks is None:
            peaks = []

        nodes = []

        for index, peak in enumerate(peaks):

            if not isinstance(peak, dict):
                continue

            z = peak.get("z")
            y = peak.get("y")
            x = peak.get("x")

            if None in (z, y, x):
                continue

            if not (
                0 <= z < heatmap.shape[0]
                and 0 <= y < heatmap.shape[1]
                and 0 <= x < heatmap.shape[2]
            ):
                continue

            try:
                node = Node(

                    node_id=index,

                    dataset=dataset,

                    t=timepoint,

                    z=float(

                        z + offsets[0, z, y, x]

                    ),

                    y=float(

                        y + offsets[1, z, y, x]

                    ),

                    x=float(

                        x + offsets[2, z, y, x]

                    ),

                    confidence=float(

                        confidence[z, y, x]

                    ),

                    embedding=embedding[
                        :,
                        z,
                        y,
                        x
                    ].tolist(),

                    division_probability=float(

                        division[z, y, x]

                    )

                )
            except (IndexError, ValueError, TypeError):
                continue

            node.radius = float(radius[z, y, x])
            nodes.append(node)

        return nodes
