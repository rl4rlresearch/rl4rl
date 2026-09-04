MECHANISM: Capacity-efficient batch-normalized CNN with higher update density

HYPOTHESIS: A deeper 216,538-parameter CNN trained with 128-image batches, mild flip augmentation, and warmup-cosine AdamW will exceed the current 8,928 validation-correct result.

INTENDED_EDIT: Replace the shallow network with four convolutional stages, halve batch size, add horizontal-flip augmentation, reduce label smoothing, and use a warmup-cosine learning-rate schedule with a nonzero floor.

EVIDENCE: The starting 105,866-parameter model reaches 89.28% after only 392 optimizer steps, leaving substantial parameter capacity and suggesting that richer spatial features plus twice as many updates can improve the fixed-exposure result.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0
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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
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
        lr=5.0e-4,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
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
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    progress = step / max(total_steps, 1)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE