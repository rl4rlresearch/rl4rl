MECHANISM: Channel-adaptive mixed-statistic downsampling

HYPOTHESIS: Replacing hard max pooling with learned per-channel mixtures of max and average pooling will exceed 9,133 correct predictions by preserving both salient edges and regional occupancy across small translations.

INTENDED_EDIT: Add a 144-parameter mixed-pooling module initialized to 75% max and 25% average, then use it at all three downsampling stages; total learned parameters become 245,178.

EVIDENCE: Translation augmentation and multi-view inference underpin the strongest design, indicating sensitivity to spatial phase, while logit calibration improved cross-entropy but remained capped at 9,133 correct. This challenges the old assumption that every channel should retain only its local maximum and instead learns whether extrema or distributed activation is more useful at each stage.

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
            torch.full((channels,), math.log(3.0))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        max_pooled = F.max_pool2d(inputs, kernel_size=2, stride=2)
        mean_pooled = F.avg_pool2d(inputs, kernel_size=2, stride=2)
        max_weight = torch.sigmoid(self.mix_logits)[None, :, None, None]
        return mean_pooled + max_weight * (max_pooled - mean_pooled)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
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
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            MixedPool2d(72),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE