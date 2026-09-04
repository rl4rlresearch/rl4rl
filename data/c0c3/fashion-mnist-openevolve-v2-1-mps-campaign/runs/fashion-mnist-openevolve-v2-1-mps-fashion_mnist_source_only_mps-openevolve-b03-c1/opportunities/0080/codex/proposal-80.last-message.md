MECHANISM: Channel-adaptive mixed-statistic downsampling

HYPOTHESIS: Replacing fixed maximum pooling with learned per-channel 80/20 max–mean pooling will exceed 9,166 correct predictions by preserving distributed shape evidence while allowing each feature channel to control its salience–smoothing tradeoff.

INTENDED_EDIT: Add a lightweight mixed-pooling module and use it at all three downsampling stages, adding 144 parameters for a total of 249,933.

EVIDENCE: Learning the previously fixed derivative filters improved correctness from 9,162 to 9,166, while repeated channel-attention refinements regressed or timed out; this challenges the shared assumption that fixed maximum-only spatial reduction is optimal.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix_logits = nn.Parameter(
            torch.full((channels,), math.log(4.0))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        maximum_weight = torch.sigmoid(self.mix_logits).view(1, -1, 1, 1)
        return maximum_weight * maximum + (1.0 - maximum_weight) * average


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
=======
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
=======
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.BatchNorm2d(72),
            nn.GELU(),
            MixedPool2d(72),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE