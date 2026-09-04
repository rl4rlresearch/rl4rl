MECHANISM: Whole-map depthwise residual refinement

HYPOTHESIS: A single 7×7 depthwise-separable residual block will exceed 9,268 correct predictions by extending the successful single-block receptive-field trend across the entire late 7×7 feature map while preserving the 56-unit spatial head.

INTENDED_EDIT: Add one 7×7 depthwise-separable residual refinement block after the final pooling layer, producing 248,914 learned parameters.

EVIDENCE: Expanding the single refinement block from 3×3 to 5×5 improved correctness from 9,250 to 9,268, whereas stacking two 3×3 blocks fell to 9,245; this supports increasing spatial context without adding depth.

<<<<<<< SEARCH
WARMUP_FRACTION = 0.08


class ImageClassifier(nn.Module):
=======
WARMUP_FRACTION = 0.08


class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(images + refined)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
>>>>>>> REPLACE