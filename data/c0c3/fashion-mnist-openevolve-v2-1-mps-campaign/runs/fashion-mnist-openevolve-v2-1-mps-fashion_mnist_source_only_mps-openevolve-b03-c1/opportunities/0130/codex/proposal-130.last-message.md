MECHANISM: Channel-adaptive mixed downsampling

HYPOTHESIS: Replacing fixed winner-take-all pooling with learned per-channel max/mean mixtures will exceed 9,210 correct predictions by reducing one-pixel aliasing while preserving the successful position-sensitive classifier.

INTENDED_EDIT: Replace all three max-pooling stages with channel-specific learned mixed pooling initialized at 90% max and 10% average; this adds 144 parameters for a total of 249,933.

EVIDENCE: Cardinal-only augmentation improved correctness from 9,196 to 9,206, and arithmetic multi-view pooling further improved it to 9,210, indicating persistent alignment sensitivity. Global spatial classification previously regressed to 9,128, so this retains the flattened spatial head while challenging the load-bearing assumption that every downsampling stage should discard all but its local maximum.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ChannelMixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.average_logit = nn.Parameter(
            torch.full((1, channels, 1, 1), -2.1972245773362196)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(inputs, kernel_size=2, stride=2)
        average = F.avg_pool2d(inputs, kernel_size=2, stride=2)
        average_weight = torch.sigmoid(self.average_logit)
        return maximum + average_weight * (average - maximum)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            ChannelMixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            ChannelMixedPool2d(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            ChannelMixedPool2d(72),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE