MECHANISM: Learned channel-wise anti-alias pooling

HYPOTHESIS: Replacing hard max pooling with learned channel-wise max/average mixtures will exceed 9,210 correct predictions by reducing phase-sensitive information loss during downsampling while retaining the successful spatial classifier.

INTENDED_EDIT: Add 144 trainable pooling logits and replace all three max-pooling layers with channel-wise mixtures initialized at 90% max and 10% average; the model remains under the 250,000-parameter ceiling.

EVIDENCE: Multi-view pooling and translation augmentation improved the available design to 9,210 correct, indicating unresolved shift sensitivity, whereas replacing spatial structure with global context and adding axial profiles fell to 9,128 and 9,167. This challenges the load-bearing assumption that repeated hard maxima are the right downsampling rule without discarding the successful flattened spatial head.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class LearnedMixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.average_logit = nn.Parameter(
            torch.full((channels,), -2.1972245773362196)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        average_weight = self.average_logit.sigmoid()[None, :, None, None]
        return maximum + average_weight * (average - maximum)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            LearnedMixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            LearnedMixedPool2d(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            LearnedMixedPool2d(72),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE