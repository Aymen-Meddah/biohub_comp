import numpy as np

from src.inference.decoder import CellDecoder


class DummyPeakFinder:
    def __init__(self, threshold=0.30):
        self.threshold = threshold

    def find(self, heatmap):
        return []


def test_decode_returns_empty_nodes_for_no_peaks(monkeypatch):
    decoder = CellDecoder(threshold=0.3)
    decoder.peak_finder = DummyPeakFinder(threshold=0.3)

    outputs = {
        "heatmap": np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
        "offsets": np.zeros((3, 1, 2, 2), dtype=np.float32),
        "radius": np.zeros((1, 1, 2, 2), dtype=np.float32),
        "embedding": np.zeros((4, 1, 2, 2), dtype=np.float32),
        "division": np.zeros((1, 1, 2, 2), dtype=np.float32),
        "confidence": np.zeros((1, 1, 2, 2), dtype=np.float32),
    }

    nodes = decoder.decode(outputs, dataset="demo", timepoint=0)

    assert nodes == []
