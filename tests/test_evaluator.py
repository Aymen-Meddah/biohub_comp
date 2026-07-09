import unittest

import torch

from src.validation.evaluator import Evaluator


class DummyModel:
    def eval(self):
        return None

    def __call__(self, images):
        return images


class DummyLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class EvaluatorTests(unittest.TestCase):

    def test_evaluate_aggregates_counts_before_computing_metrics(self):
        evaluator = Evaluator(model=DummyModel(), device="cpu")
        evaluator.decoder.decode = lambda outputs, dataset, timepoint: []
        evaluator.matcher.match = lambda predictions, targets: ([], [], [])

        results = iter([
            {"tp": 1, "fp": 2, "fn": 0, "precision": 1 / 3, "recall": 1.0, "f1": 0.5},
            {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3},
        ])
        evaluator.metrics.compute = lambda matches, false_positive, false_negative: next(results)

        batches = [
            {"image": torch.zeros(1, 1, 1, 1), "dataset": ["sample"], "timepoint": [0], "cells": [[]]},
            {"image": torch.zeros(1, 1, 1, 1), "dataset": ["sample"], "timepoint": [0], "cells": [[]]},
        ]

        metrics = evaluator.evaluate(DummyLoader(batches))

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.5714285714)


if __name__ == "__main__":
    unittest.main()
