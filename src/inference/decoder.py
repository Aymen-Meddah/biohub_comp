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

        required_keys = {
            "heatmap",
            "offsets",
            "radius",
            "embedding",
            "division",
            "confidence",
        }
        if not required_keys.issubset(outputs.keys()):
            return []

        try:
            heatmap = torch.sigmoid(outputs["heatmap"][0, 0]).detach().cpu().numpy()
            offsets = outputs["offsets"][0].detach().cpu().numpy()
            radius = torch.sigmoid(outputs["radius"][0, 0]).detach().cpu().numpy()
            embedding = outputs["embedding"][0].detach().cpu().numpy()
            division = torch.sigmoid(outputs["division"][0, 0]).detach().cpu().numpy()
            confidence = torch.sigmoid(outputs["confidence"][0, 0]).detach().cpu().numpy()
        except Exception:
            return []

        if heatmap.ndim != 3:
            return []

        peaks = self.peak_finder.find(heatmap)
        if not peaks:
            return []

        nodes = []

        for index, peak in enumerate(peaks):
            if not isinstance(peak, dict):
                continue

            try:
                z = int(peak["z"])
                y = int(peak["y"])
                x = int(peak["x"])
            except (KeyError, TypeError, ValueError):
                continue

            if not (
                0 <= z < heatmap.shape[0]
                and 0 <= y < heatmap.shape[1]
                and 0 <= x < heatmap.shape[2]
            ):
                continue

            try:
                offset_z = float(offsets[0, z, y, x])
                offset_y = float(offsets[1, z, y, x])
                offset_x = float(offsets[2, z, y, x])
                confidence_score = float(confidence[z, y, x])
                division_probability = float(division[z, y, x])
                radius_value = float(radius[z, y, x])
                embedding_vector = embedding[:, z, y, x].tolist()
            except (IndexError, ValueError, TypeError):
                continue

            node = Node(
                node_id=index,
                dataset=dataset,
                t=timepoint,
                z=z + offset_z,
                y=y + offset_y,
                x=x + offset_x,
                confidence=confidence_score,
                embedding=embedding_vector,
                division_probability=division_probability,
            )
            node.radius = radius_value
            nodes.append(node)

        return nodes
