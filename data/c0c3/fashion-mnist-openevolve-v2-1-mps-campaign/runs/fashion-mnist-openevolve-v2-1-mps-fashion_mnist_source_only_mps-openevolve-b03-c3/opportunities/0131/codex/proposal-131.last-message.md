MECHANISM: Channelwise learnable max–average pooling

HYPOTHESIS: Extending the 9,284-correct mixed-pooling design from one mixture coefficient per stage to one per feature channel will exceed 9,284 correct by letting edge-selective and region-selective channels choose different pooling behavior with only 72 additional parameters.

INTENDED_EDIT: Restore beta2=0.96 and replace both hard max-pooling layers with 90%-max channelwise learnable mixtures.

EVIDENCE: Scalar mixed pooling improved the best verified result to 9,284 correct with lower cross-entropy, while larger shared-capacity additions regressed; a channelwise extension preserves that successful mechanism with minimal capacity and runtime overhead.

<<<<<<< SEARCH
from torch.nn import functional as F

BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
from torch.nn import functional as F

BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ChannelwiseMixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_logits = nn.Parameter(
            torch.full((channels,), 2.1972246)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weights = torch.sigmoid(self.max_logits)[None, :, None, None]
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weights * maximum + (1.0 - max_weights) * average


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            ChannelwiseMixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ChannelwiseMixedPool2d(48),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
        betas=(0.9, 0.97),
=======
        betas=(0.9, 0.96),
>>>>>>> REPLACE