MECHANISM: Bottleneck residual spatial refinement

HYPOTHESIS: Reallocating 6,752 parameters to a residual 7×7 feature-refinement block while slightly narrowing the dense head will exceed 9,311 correct predictions without exceeding the parameter ceiling.

INTENDED_EDIT: Add a lightweight pointwise–depthwise–pointwise residual block before the final pooling layer and reduce the classifier width from 147 to 139.

EVIDENCE: Evaluation-temperature tuning plateaued at exactly 9,311 correct, including identical results at 0.9350 and 0.9351, so improving the primary objective requires changing learned predictions; this targeted reallocation tests residual spatial refinement without the complexity of the unverified multiscale redesign.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 1.0


class ResidualRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(96, 32, kernel_size=1, bias=False)
        self.reduce_norm = nn.BatchNorm2d(32)
        self.depthwise = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, groups=32, bias=False
        )
        self.depthwise_norm = nn.BatchNorm2d(32)
        self.expand = nn.Conv2d(32, 96, kernel_size=1, bias=False)
        self.expand_norm = nn.BatchNorm2d(96)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.reduce_norm(self.reduce(features)))
        residual = F.gelu(self.depthwise_norm(self.depthwise(residual)))
        residual = self.expand_norm(self.expand(residual))
        return features + residual


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
            ResidualRefinement(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
=======
            nn.Linear(96 * 3 * 3, 139),
            nn.BatchNorm1d(139),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(139, 10),
>>>>>>> REPLACE