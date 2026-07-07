from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from config import Config
from src.data.index_builder import IndexBuilder
from src.data.zarr_reader import ZarrReader
from src.data.geff_reader import GeffReader
from src.data.patch_sampler import PatchSampler

from src.data.target_builder import TargetBuilder

from src.data.heatmap_builder import HeatmapBuilder
from src.data.offset_builder import OffsetBuilder
from src.data.radius_builder import RadiusBuilder
from src.data.division_builder import DivisionBuilder
from src.data.embedding_builder import EmbeddingBuilder


class BioHubDataset(Dataset):

    def __init__(
        self,
        data_dir=None,
        train_dir=None,
        split="train",
        validation_fraction=0.2,
        patch_size=(32, 96, 96),
        sigma=2.5,
        embedding_dim=16,
        zarr_array_key=None
    ):

        if train_dir is not None and data_dir is None:
            data_dir = train_dir
        if data_dir is None:
            data_dir = Config.TRAIN_DIR if split != "test" else Config.TEST_DIR

        self.data_dir = Path(data_dir)
        self.split = split
        self.zarr_array_key = zarr_array_key

        samples = IndexBuilder(self.data_dir).build()
        self.samples = self._split_samples(
            samples,
            split,
            validation_fraction
        )

        self.patch_sampler = PatchSampler(
            patch_size
        )

        self.target_builder = TargetBuilder()

        self.heatmap_builder = HeatmapBuilder(
            sigma=sigma
        )

        self.offset_builder = OffsetBuilder()

        self.radius_builder = RadiusBuilder()

        self.division_builder = DivisionBuilder()

        self.embedding_builder = EmbeddingBuilder(
            embedding_dim
        )

    @staticmethod
    def _split_samples(samples, split, validation_fraction):
        if split not in {"train", "validation", "val", "test", "all"}:
            raise ValueError(f"Unknown split: {split}")
        if split in {"test", "all"}:
            return samples

        if len(samples) <= 1:
            return samples

        validation_count = max(1, int(round(len(samples) * validation_fraction)))
        validation_count = min(validation_count, len(samples) - 1)

        if split in {"validation", "val"}:
            return samples[-validation_count:]
        return samples[:-validation_count]

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        zarr_path = sample.get("zarr")
        if not zarr_path:
            raise FileNotFoundError(
                f"Sample '{sample.get('dataset', 'unknown')}' is missing a zarr path. "
                f"Check the dataset root: {self.data_dir}"
            )

        volume = ZarrReader(
            zarr_path,
            array_key=self.zarr_array_key
        )

        graph = GeffReader(
            sample["geff"]
        )

        patch = self.patch_sampler.random_patch(
            volume.shape
        )

        image = self.patch_sampler.extract(
            volume,
            patch
        )
        image = self._normalize_image(image)

        nodes = graph.nodes_at_time(
            patch["t"]
        )

        cells = self.target_builder.cells_in_patch(
            nodes,
            patch
        )

        spatial_shape = image.shape[-3:]

        heatmap = self.heatmap_builder.generate(
            spatial_shape,
            cells
        )

        offset = self.offset_builder.generate(
            spatial_shape,
            cells
        )

        radius = self.radius_builder.generate(
            spatial_shape,
            cells
        )

        division = self.division_builder.generate(
            spatial_shape,
            cells
        )

        embedding = self.embedding_builder.generate(
            spatial_shape,
            cells
        )
        confidence = np.ones_like(
            heatmap,
            dtype=np.float32
        )
        targets = {
            "heatmap": heatmap[None, ...],
            "offset": offset,
            "radius": radius[None, ...],
            "division": division[None, ...],
            "embedding": embedding,
            "confidence": confidence[None, ...],
        }

        return {

            "image": image,

            "heatmap": targets["heatmap"],

            "offset": targets["offset"],

            "radius": targets["radius"],

            "division": targets["division"],

            "embedding": targets["embedding"],

            "cells": cells,

            "dataset": sample["dataset"],

            "patch": patch,

            "timepoint": patch["t"],

            "confidence": targets["confidence"],

            "targets": targets,
        }

    @staticmethod
    def _normalize_image(image):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3:
            image = image[None, ...]
        max_value = float(np.max(image))
        if max_value > 0:
            image = image / max_value
        return image


CellTrackingDataset = BioHubDataset
