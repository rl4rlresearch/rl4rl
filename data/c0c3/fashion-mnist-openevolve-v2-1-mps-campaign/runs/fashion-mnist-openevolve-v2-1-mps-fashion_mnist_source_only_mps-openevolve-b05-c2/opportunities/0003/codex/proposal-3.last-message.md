MECHANISM: Matched translation-and-flip test-time ensembling

HYPOTHESIS: Adding four one-pixel translation views to the best 249,260-parameter residual design’s flip ensemble will exceed 9,177 correct predictions by reducing residual shift sensitivity without changing training exposure or learned-parameter count.

INTENDED_EDIT: Adopt Reference Design 2’s residual CNN and training recipe, then average validation logits across centered, four cardinally shifted, and horizontally flipped views.

EVIDENCE: Reference Design 2 achieved 9,177 correct with random ±2-pixel translations but evaluates only the centered image and its flip, motivating an evaluation ensemble matched to its learned translation invariance.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
PEAK_LR = 2.0e-3
MIN_LR = 2.0e-5
>>>>>>> REPLACE

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.GELU(),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
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
            padded[:, :, 0:height, 1 : width + 1],
            padded[:, :, 2 : height + 2, 1 : width + 1],
            padded[:, :, 1 : height + 1, 0:width],
            padded[:, :, 1 : height + 1, 2 : width + 2],
        )
        for view in shifted_views:
            logit_sum = logit_sum + self._forward_once(view)
            logit_sum = logit_sum + self._forward_once(view.flip(-1))
        return logit_sum / 10.0
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=2e-4)
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps

    batch_size, _, height, width = images.shape
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
    images = padded[
        batch_index, 0, row_index, column_index
    ].unsqueeze(1)

    flip_mask = torch.rand(batch_size, device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = MIN_LR + (PEAK_LR - MIN_LR) * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE