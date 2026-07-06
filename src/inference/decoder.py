import numpy as np
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

        heatmap = outputs["heatmap"][0, 0]

        offsets = outputs["offsets"][0]

        radius = outputs["radius"][0, 0]

        embedding = outputs["embedding"][0]

        division = outputs["division"][0, 0]

        confidence = outputs["confidence"][0, 0]

        peaks = self.peak_finder.find(

            heatmap

        )

        nodes = []

        for index, peak in enumerate(peaks):

            z = peak["z"]

            y = peak["y"]

            x = peak["x"]

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

        node.radius = float(

            radius[z, y, x]

        )

        nodes.append(node)

        return nodes