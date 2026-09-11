MECHANISM: Pre-pooling residual depthwise refinement

HYPOTHESIS: Adding a low-cost residual spatial/channel refinement block before the unchanged terminal pool will exceed 9,286 validation-correct predictions while remaining below 250,000 parameters and adding little runtime.

INTENDED_EDIT: Add a 10,464-parameter depthwise-separable residual block at the 7×7 feature stage, preserving the proven optimizer, augmentation, pooling, averaging, and TTA procedure.

EVIDENCE: Changing terminal pooling regressed from 9,286 to 9,234, while larger representation redesigns could not be verified; this tests additional representation capacity without disturbing the successful pooling path or adding a parallel branch.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ResidualRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        refined = F.gelu(self.depthwise_norm(self.depthwise(features)))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(features + refined)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualRefinement(96),
            nn.MaxPool2d(2),
>>>>>>> REPLACE