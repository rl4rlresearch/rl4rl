MECHANISM: Late depthwise-separable spatial residual refinement

HYPOTHESIS: Adding an inexpensive 7×7 spatial refinement block to the best multi-scale model will exceed 9,202 correct predictions by improving local feature interactions that global channel gating could not capture.

INTENDED_EDIT: Add a residual depthwise/pointwise convolutional block after late multi-scale fusion and reduce the classifier width from 64 to 61, yielding 249,961 learned parameters.

EVIDENCE: Multi-scale spatial representation raised correctness from 9,112 to 9,202, while spending the remaining capacity on channel gating tied at 9,202 with worse cross-entropy; this motivates using the capacity for additional spatial refinement instead.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 10),
        )
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
=======
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
>>>>>>> REPLACE