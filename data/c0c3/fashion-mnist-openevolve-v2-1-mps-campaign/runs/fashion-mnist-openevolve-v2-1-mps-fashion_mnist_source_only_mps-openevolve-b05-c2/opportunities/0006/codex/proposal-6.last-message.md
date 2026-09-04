MECHANISM: Matched one-pixel translation invariance

HYPOTHESIS: Using the best residual 3×3 TTA design while restricting training translations to the same ±1-pixel support will exceed 9,191 correct predictions by avoiding harmful extreme shifts.

INTENDED_EDIT: Replace the current CNN with the 249,260-parameter residual network and full 3×3 translation/flip ensemble, then narrow random training translations from ±2 to ±1 pixels.

EVIDENCE: The residual design with uniform 3×3 TTA achieved 9,191 correct, while expanding evaluation to the full ±2 support fell to 9,188, suggesting two-pixel shifts can dilute useful predictions and motivating matched narrower augmentation.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.BatchNorm1d(56),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(56, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(images.flip(-1))
        return 0.5 * (logits + flipped_logits)
=======
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + residual, inplace=True)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 96),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 42),
            nn.BatchNorm1d(42),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(42, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        logit_sum = logits + self._forward_once(images.flip(-1))
        _, _, height, width = images.shape
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:height, 0:width],
            padded[:, :, 0:height, 1 : width + 1],
            padded[:, :, 0:height, 2 : width + 2],
            padded[:, :, 1 : height + 1, 0:width],
            padded[:, :, 1 : height + 1, 2 : width + 2],
            padded[:, :, 2 : height + 2, 0:width],
            padded[:, :, 2 : height + 2, 1 : width + 1],
            padded[:, :, 2 : height + 2, 2 : width + 2],
        )
        for view in shifted_views:
            logit_sum = logit_sum + self._forward_once(view)
            logit_sum = logit_sum + self._forward_once(view.flip(-1))
        return logit_sum / 18.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_index = torch.arange(
        batch_size, device=images.device
    )[:, None, None]
    row_index = (
        torch.arange(height, device=images.device)[None, :, None]
        + torch.randint(0, 5, (batch_size, 1, 1), device=images.device)
    )
    column_index = (
        torch.arange(width, device=images.device)[None, None, :]
        + torch.randint(0, 5, (batch_size, 1, 1), device=images.device)
    )
=======
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    batch_index = torch.arange(
        batch_size, device=images.device
    )[:, None, None]
    row_index = (
        torch.arange(height, device=images.device)[None, :, None]
        + torch.randint(0, 3, (batch_size, 1, 1), device=images.device)
    )
    column_index = (
        torch.arange(width, device=images.device)[None, None, :]
        + torch.randint(0, 3, (batch_size, 1, 1), device=images.device)
    )
>>>>>>> REPLACE