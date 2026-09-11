MECHANISM: Batch-normalized deeper CNN with per-image geometric augmentation and flip-ensemble inference

HYPOTHESIS: Increasing useful capacity to 216,346 parameters, doubling optimizer updates, and training translation/flip invariance will exceed the current 8,928 correct predictions after the same 100,000-example exposure.

INTENDED_EDIT: Replace the shallow CNN with a four-convolution network, use batch size 128, per-image random translations and flips, evaluation-time flip averaging, decoupled AdamW parameter groups, and a warmup-cosine schedule.

EVIDENCE: The starting 105,866-parameter two-layer CNN reaches 8,928 correct with only 392 optimizer steps; substantial parameter headroom and the short fixed exposure motivate a higher-capacity normalized model with more update opportunities.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0
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
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=4e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    batch_index = torch.arange(images.shape[0], device=images.device)
    shifts = torch.randint(0, 5, (images.shape[0], 2), device=images.device)
    images = windows[batch_index, :, shifts[:, 0], shifts[:, 1]]
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
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
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    warmup_fraction = 0.08
    start_lr = 4e-4
    peak_lr = 2.5e-3
    end_lr = 1e-4
    if progress < warmup_fraction:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup_fraction
    else:
        cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        lr = end_lr + (peak_lr - end_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
>>>>>>> REPLACE