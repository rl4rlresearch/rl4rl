MECHANISM: Per-example random translation augmentation

HYPOTHESIS: Training the verified 9,273-correct pairwise-refinement model with independent zero-padded translations of up to two pixels will exceed 9,273 correct predictions by learning modest positional invariance without changing model capacity or the successful optimization schedule.

INTENDED_EDIT: Restore the best 248,426-parameter design and its 1.17817 evaluation calibration, then apply a random 28×28 crop from images padded by two pixels during training.

EVIDENCE: Reference Design 2 achieved the best validation result at 9,273 correct; architectural refinements, EMA, alternate pooling, loss annealing, and dropout removal all regressed, motivating an isolated input-regularization change while preserving every verified model and optimizer setting.

<<<<<<< SEARCH
BATCH_SIZE = 256
=======
BATCH_SIZE = 64
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
class PairwiseRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            128, 128, kernel_size=5, padding=2, groups=64, bias=False
        )
        self.norm = nn.BatchNorm2d(128)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(features)))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            PairwiseRefinement(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = 1.17817 * logits
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if parameter.ndim == 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.5e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2))
    crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offsets_y = torch.randint(0, 5, (images.shape[0],), device=images.device)
    offsets_x = torch.randint(0, 5, (images.shape[0],), device=images.device)
    images = crops[
        batch_indices, :, offsets_y, offsets_x
    ].contiguous()
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
    peak_lr = 2.5e-3
    minimum_lr = 1.0e-4
    completed = min(step + 1, max(total_steps, 1))
    warmup_steps = max(1, int(0.05 * total_steps))
    if completed <= warmup_steps:
        fraction = completed / warmup_steps
        learning_rate = peak_lr * (0.1 + 0.9 * fraction)
    else:
        fraction = (completed - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        learning_rate = minimum_lr + (peak_lr - minimum_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE