MECHANISM: Learnable max-average downsampling

HYPOTHESIS: Replacing hard max pooling with a learnable max/average mixture will exceed 9,257 correct predictions by reducing shift aliasing while retaining salient garment features.

INTENDED_EDIT: Add a two-parameter mixed-pooling module, initialized 75% toward max pooling, and use it at both downsampling stages; total parameters become 233,436.

EVIDENCE: Evaluation-only crop weighting improved correctness, showing sensitivity to spatial shifts, while classifier dropout and posterior-space aggregation did not improve the widened model; this tests feature-level shift robustness without adding depth or significant capacity.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(math.log(3.0)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        return (
            max_weight * F.max_pool2d(inputs, 2)
            + (1.0 - max_weight) * F.avg_pool2d(inputs, 2)
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(16, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            MixedPool2d(),
        )
>>>>>>> REPLACE