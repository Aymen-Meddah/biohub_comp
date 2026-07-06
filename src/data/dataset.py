from torch.utils.data import Dataset

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
        train_dir,
        patch_size=(32, 96, 96),
        sigma=2.5,
        embedding_dim=16
    ):

        self.samples = IndexBuilder(
            train_dir
        ).build()

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

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        volume = ZarrReader(
            sample["zarr"]
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

        nodes = graph.nodes_at_time(
            patch["t"]
        )

        cells = self.target_builder.cells_in_patch(
            nodes,
            patch
        )

        heatmap = self.heatmap_builder.generate(
            image.shape,
            cells
        )

        offset = self.offset_builder.generate(
            image.shape,
            cells
        )

        radius = self.radius_builder.generate(
            image.shape,
            cells
        )

        division = self.division_builder.generate(
            image.shape,
            cells
        )

        embedding = self.embedding_builder.generate(
            image.shape,
            cells
        )
        confidence = np.ones_like(
            heatmap,
            dtype=np.float32
        )
        return {

            "image": image,

            "heatmap": heatmap,

            "offset": offset,

            "radius": radius,

            "division": division,

            "embedding": embedding,

            "cells": cells,

            "dataset": sample["dataset"],

            "patch": patch,

            "confidence": confidence,
        }