MECHANISM: Learnable max–average mixed pooling

HYPOTHESIS: Replacing hard max pooling with a 90%-max learnable mixture will exceed 9,283 correct predictions by reducing translation aliasing while preserving the baseline’s edge-selective behavior.

INTENDED_EDIT: Add a two-parameter mixed-pooling module and use it at both downsampling stages; preserve all other architecture, optimization, loss, augmentation, schedule, and TTA settings.

EVIDENCE: The 9,283-correct baseline depends heavily on translation augmentation and TTA, while changing the translation kernel or probability-space marginalization regressed. This motivates improving translation stability inside the representation with minimal added capacity.

<<<<<<< SEARCH
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE