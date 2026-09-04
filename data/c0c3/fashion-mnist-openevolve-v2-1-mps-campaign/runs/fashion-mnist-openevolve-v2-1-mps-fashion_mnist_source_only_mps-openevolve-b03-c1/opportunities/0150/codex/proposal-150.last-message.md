MECHANISM: Channel-adaptive mixed downsampling

HYPOTHESIS: Replacing fixed max pooling with learned per-channel max/mean mixtures will exceed 9,210 correct predictions by reducing shift-sensitive aliasing while preserving strong local responses.

INTENDED_EDIT: Challenge the assumption that every channel benefits from hard max pooling. Add 144 learned pooling logits, initialized to 90% maximum and 10% mean, while retaining the successful position-sensitive classifier.

EVIDENCE: Cardinal augmentation improved correctness from 9,196 to 9,206 and multi-view pooling reached 9,210, indicating unresolved alignment sensitivity; global spatial classification fell to 9,128, motivating smoother learned downsampling without discarding the flattened spatial head.

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
        self.maximum_logit = nn.Parameter(
            torch.full((channels,), math.log(9.0))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(inputs, kernel_size=2, stride=2)
        average = F.avg_pool2d(inputs, kernel_size=2, stride=2)
        maximum_fraction = torch.sigmoid(self.maximum_logit)[
            None, :, None, None
        ]
        return average + maximum_fraction * (maximum - average)


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