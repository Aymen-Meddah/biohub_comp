import torch 
import numpy as np 


TARGET_KEYS = (
    "heatmap",
    "offset",
    "radius",
    "division",
    "embedding",
    "confidence",
)


def to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    return torch.tensor(value, dtype=torch.float32)


def collate_fn(batch):
    images = torch.stack([
        to_tensor(sample["image"])
        for sample in batch
    ])

    targets = {
        key: torch.stack([
            to_tensor(sample[key])
            for sample in batch
        ])
        for key in TARGET_KEYS
    }

    return {
        "image": images,

        **targets,

        "targets": targets,

        "dataset": [sample["dataset"] for sample in batch],

        "patch": [sample["patch"] for sample in batch],

        "timepoint": torch.tensor(
            [sample["timepoint"] for sample in batch],
            dtype=torch.long
        ),

        "cells": [sample["cells"] for sample in batch],

    }
