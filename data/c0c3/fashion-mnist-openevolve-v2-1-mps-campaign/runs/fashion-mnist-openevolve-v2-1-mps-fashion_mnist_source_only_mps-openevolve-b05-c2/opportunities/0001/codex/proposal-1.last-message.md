MECHANISM: Invariance-trained deeper batch-normalized CNN with flip test-time averaging

HYPOTHESIS: Increasing convolutional capacity while training on random translations and flips will exceed 8,928 correct predictions within the fixed exposure and parameter budgets.

INTENDED_EDIT: Use a 241,538-parameter four-convolution network, batch size 128, spatial augmentation, lighter label smoothing, cosine AdamW, and horizontal-flip evaluation averaging.

EVIDENCE: The current 105,866-parameter two-convolution model reaches 89.28% accuracy, suggesting additional feature-extraction capacity and explicit spatial invariance are promising under the 250,000-parameter ceiling.

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
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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