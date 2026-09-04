MECHANISM: Identity-initialized input-conditioned channel routing

HYPOTHESIS: Dynamically reweighting each convolutional stage from its image-wide channel context will exceed 9,286 validation-correct predictions while remaining within the parameter and runtime limits.

INTENDED_EDIT: Add lightweight squeeze-excitation gates after the 32-, 64-, and 96-channel stages; zero-initialize their final projections so the network begins as the proven static backbone and learns image-dependent channel routing.

EVIDENCE: Calibration and tail-averaging changes plateaued at 9,286 correct, while the relational-token head timed out. This challenges the shared assumption of static image-independent channel processing with only 7,408 additional parameters and negligible spatial computation.

<<<<<<< SEARCH
BATCH_SIZE = 48
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 48
GRAD_CLIP_NORM = 1.0


class ChannelGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.project = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = features.mean(dim=(2, 3))
        scale = 2.0 * torch.sigmoid(self.project(context))
        return features * scale[:, :, None, None]


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ChannelGate(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ChannelGate(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
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
            ChannelGate(96),
            nn.MaxPool2d(2),
>>>>>>> REPLACE