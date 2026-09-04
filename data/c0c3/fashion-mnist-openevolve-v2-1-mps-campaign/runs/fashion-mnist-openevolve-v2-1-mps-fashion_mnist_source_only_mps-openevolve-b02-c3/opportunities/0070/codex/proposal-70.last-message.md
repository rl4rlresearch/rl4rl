MECHANISM: Deep residual part reasoning with position-sensitive class readout

HYPOTHESIS: Reallocating the oversized nonlinear dense head into four spatial residual bottlenecks will exceed 9,344 correct predictions by learning richer cross-channel part relationships while retaining the spatial organization that global pooling lost.

INTENDED_EDIT: Replace the assumption that a shallow feature extractor needs a 176K-parameter shared dense bottleneck with four local/contextual residual reasoning blocks and a direct class-specific 7×7 spatial readout; retain the proven refinement gate, training procedure, TTA, and EMA.

EVIDENCE: Global pooling fell to 9,085 correct and post-pooling attention reached only 9,300, showing that spatial organization and pre-prediction feature formation are load-bearing; this design preserves position explicitly while moving capacity from the current dense head into learned spatial computation.

<<<<<<< SEARCH
        return F.gelu(images + gate * refined)


class ImageClassifier(nn.Module):
=======
        return F.gelu(images + gate * refined)


class SpatialReasoningBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.reduce_norm = nn.BatchNorm2d(hidden_channels)
        self.spatial = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
            padding_mode="replicate",
        )
        self.spatial_norm = nn.BatchNorm2d(hidden_channels)
        self.expand = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.expand_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.reduce_norm(self.reduce(images)))
        residual = F.gelu(self.spatial_norm(self.spatial(residual)))
        residual = self.expand_norm(self.expand(residual))
        return images + 0.5 * residual


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialReasoningBlock(64, 56, dilation=1),
            SpatialReasoningBlock(64, 56, dilation=2),
            SpatialReasoningBlock(64, 56, dilation=1),
            SpatialReasoningBlock(64, 56, dilation=2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.10),
            nn.Linear(64 * 7 * 7, 10),
        )
>>>>>>> REPLACE