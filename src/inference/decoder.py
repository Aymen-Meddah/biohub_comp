import numpy as np
import torch
from src.tracking.node import Node
from src.inference.peak_finder import PeakFinder


class CellDecoder:

    def __init__(

        self,

        threshold=0.30,
        max_detections=100,

    ):

        self.peak_finder = PeakFinder(

            threshold=threshold

        )
        # minimum combined score threshold (heatmap * confidence)
        self.threshold = float(threshold)
        # max detections per frame to limit false positives
        self.max_detections = int(max_detections)

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
            # keep tensors where convenient for potential further ops
            heatmap_t = torch.sigmoid(outputs["heatmap"][0, 0])
            heatmap = heatmap_t.detach().cpu().numpy()
            offsets = outputs["offsets"][0].detach().cpu().numpy()
            radius = torch.sigmoid(outputs["radius"][0, 0]).detach().cpu().numpy()
            embedding = outputs["embedding"][0].detach().cpu().numpy()
            division = torch.sigmoid(outputs["division"][0, 0]).detach().cpu().numpy()
            confidence_t = torch.sigmoid(outputs["confidence"][0, 0])
            confidence = confidence_t.detach().cpu().numpy()
        except Exception:
            return []

        if heatmap.ndim != 3:
            return []

        # find local peaks in the heatmap (NMS + thresholding)
        peaks = self.peak_finder.find(heatmap)
        if not peaks:
            return []

        # filter by combined score (heatmap score * confidence) and keep top-K
        for p in peaks:
            z = int(p["z"]) ; y = int(p["y"]) ; x = int(p["x"])
            # safety check in case shapes mismatch
            try:
                conf = float(confidence[z, y, x])
            except Exception:
                conf = 0.0
            p["confidence"] = conf
            p["combined_score"] = float(p["score"]) * conf

        # reject by combined score threshold
        peaks = [p for p in peaks if p.get("combined_score", 0.0) >= self.threshold]
        if not peaks:
            return []

        # sort by combined score and keep top detections
        peaks.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        peaks = peaks[: self.max_detections]

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
