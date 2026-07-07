import torch
import torch.nn as nn

from src.models.encoder import Encoder
from src.models.decoder import Decoder

from src.models.heads import (
    HeatmapHead,
    OffsetHead,
    RadiusHead,
    DivisionHead,
    EmbeddingHead,
    ConfidenceHead
)

class BioHubModel(nn.Module):

    def __init__(self):

        super().__init__()

        ####################################################
        # Backbone
        ####################################################

        self.encoder = Encoder()

        self.decoder = Decoder()

        ####################################################
        # Heads
        ####################################################

        self.heatmap_head = HeatmapHead(16)

        self.offset_head = OffsetHead(16)

        self.radius_head = RadiusHead(16)

        self.division_head = DivisionHead(16)

        self.confidence_head = ConfidenceHead(16)

        self.embedding_head = EmbeddingHead(
            16,
            embedding_dim=16
        )

    def forward(self, x):

        ####################################################
        # Encoder
        ####################################################

        features = self.encoder(x)

        ####################################################
        # Decoder
        ####################################################

        decoder_features = self.decoder(features)

        ####################################################
        # Heads
        ####################################################

        heatmap = self.heatmap_head(
            decoder_features
        )

        offsets = self.offset_head(
            decoder_features
        )

        radius = self.radius_head(
    decoder_features
        )

        division = self.division_head(
            decoder_features
        )

        embedding = self.embedding_head(
            decoder_features
        )

        confidence = self.confidence_head(
    decoder_features
        )
        ####################################################
        # Output
        ####################################################

        return {

    "heatmap": heatmap,

    "offsets": offsets,

    "radius": radius,

    "division": division,

    "embedding": embedding,

    "confidence": confidence

}