MECHANISM: Orthogonal silhouette-statistic pooling

HYPOTHESIS: Replacing the parameter-heavy flattened 7×7 bottleneck with learned row/column mean-and-maximum projections plus deeper high-level spatial processing will exceed 9,251 correct predictions, or tie while reducing cross-entropy below 0.20655155601501465.

INTENDED_EDIT: Add one efficient 96-channel residual block, classify axis-wise spatial statistics through a wider nonlinear head, and retain the best verified 1.0495 inference calibration.

EVIDENCE: All completed 9,251-correct designs share the same 4,704-value flattening head, while prediction-side calibration and EMA blending have plateaued. This challenges the load-bearing assumption that most parameters should model every fixed spatial cell; axis projections instead emphasize clothing silhouettes and local feature presence while reallocating capacity to a 64-unit decision layer.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class AxisProjectionHead(nn.Module):
    def __init__(self, channels: int, hidden_features: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(channels * 7 * 4, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_features, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        vertical_mean = features.mean(dim=3)
        vertical_max = features.amax(dim=3)
        horizontal_mean = features.mean(dim=2)
        horizontal_max = features.amax(dim=2)
        projections = torch.cat(
            (vertical_mean, vertical_max, horizontal_mean, horizontal_max),
            dim=2,
        )
        return self.classifier(projections.flatten(1))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
=======
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
            ResidualDepthwiseBlock(96),
        )
        self.classifier = AxisProjectionHead(96, 64)
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.048 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE