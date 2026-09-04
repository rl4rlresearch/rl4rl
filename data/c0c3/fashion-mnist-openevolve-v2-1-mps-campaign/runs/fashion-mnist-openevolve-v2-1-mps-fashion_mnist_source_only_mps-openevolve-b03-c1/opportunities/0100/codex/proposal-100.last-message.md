MECHANISM: Per-channel learned max–average downsampling

HYPOTHESIS: Replacing fixed max pooling with learned per-channel mixtures of max and average pooling will exceed 9,166 correct predictions by retaining distributed shape evidence and providing denser encoder gradients within the short fixed-exposure budget.

INTENDED_EDIT: Add a lightweight mixed-pooling layer and replace all three fixed 2×2 max-pooling stages, adding 144 parameters for a 249,933-parameter model while preserving feature-map geometry and the existing classifier.

EVIDENCE: Head replacements and global-context classification either timed out or regressed to 9,128, while loss and TTA changes failed to beat 9,166. This challenges their shared assumption that fixed winner-take-all downsampling already provides an adequate representation, using a computationally inexpensive learned alternative initialized close to max pooling.

<<<<<<< SEARCH
from torch.nn import functional as F

BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
from torch.nn import functional as F

BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self, channels: int, initial_max_weight: float = 0.8) -> None:
        super().__init__()
        initial_logit = math.log(
            initial_max_weight / (1.0 - initial_max_weight)
        )
        self.max_logits = nn.Parameter(
            torch.full((channels,), initial_logit)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(inputs, kernel_size=2, stride=2)
        average = F.avg_pool2d(inputs, kernel_size=2, stride=2)
        maximum_weight = torch.sigmoid(self.max_logits)[None, :, None, None]
        return average + maximum_weight * (maximum - average)


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