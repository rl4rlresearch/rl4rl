MECHANISM: Augmented deep convolutional model with test-time flip ensembling

HYPOTHESIS: Increasing capacity from 105,866 to 232,682 parameters, doubling optimizer updates via batch size 128, and adding translation/flip invariance will improve beyond 8,928 correct validation predictions.

INTENDED_EDIT: Replace the shallow CNN with a five-convolution BatchNorm network, random crops and flips, evaluation-time flip averaging, and a warmup-cosine AdamW schedule.

EVIDENCE: The starting model reaches 89.28% accuracy with only two convolution layers and 392 optimizer steps, leaving substantial parameter and update capacity under the 250,000-parameter ceiling.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0


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
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0


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
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(96 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 10),
        )

    def _logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._logits(images)
        if self.training:
            return logits
        return 0.5 * (logits + self._logits(images.flip(-1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=4.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
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
    if step < int(0.85 * total_steps):
        count = images.size(0)
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        top = torch.randint(0, 5, (count,), device=images.device)
        left = torch.randint(0, 5, (count,), device=images.device)
        batch_index = torch.arange(count, device=images.device)[:, None, None]
        row_index = top[:, None, None] + torch.arange(
            28, device=images.device
        )[None, :, None]
        column_index = left[:, None, None] + torch.arange(
            28, device=images.device
        )[None, None, :]
        images = padded[
            batch_index, 0, row_index, column_index
        ].unsqueeze(1)
        flip_mask = torch.rand(
            (count, 1, 1, 1), device=images.device
        ) < 0.5
        images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    warmup = 0.08
    start_lr = 4.0e-4
    peak_lr = 3.0e-3
    end_lr = 1.5e-4
    if progress < warmup:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup
    else:
        decay_progress = (progress - warmup) / (1.0 - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        lr = end_lr + (peak_lr - end_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
>>>>>>> REPLACE