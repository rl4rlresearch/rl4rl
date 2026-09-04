MECHANISM: Per-example random translation with validation-aligned flip ensembling

HYPOTHESIS: Adding random ±2-pixel translations to the best linear flip-loss curriculum will exceed 9,330 correct predictions by improving local translation robustness without changing its proven architecture or inference rule.

INTENDED_EDIT: Adopt Reference Design 1 and apply independently sampled two-dimensional translations during batch preparation while retaining its batch size, residual model, optimizer, linear ensemble-loss curriculum, and label smoothing.

EVIDENCE: Linear ensemble-loss annealing achieved the best result of 9,330 correct, while subsequent smoothing and curriculum-shape changes fell to 9,322; this motivates preserving the winning design and testing an orthogonal, previously untested input-invariance mechanism.

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
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
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
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
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
    del step, total_steps
    padding = 2
    height, width = images.shape[-2:]
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )

    batch_size, channels = images.shape[:2]
    top = torch.randint(
        0,
        2 * padding + 1,
        (batch_size,),
        device=images.device,
    )
    left = torch.randint(
        0,
        2 * padding + 1,
        (batch_size,),
        device=images.device,
    )

    rows = torch.arange(height, device=images.device).unsqueeze(0)
    rows = rows + top.unsqueeze(1)
    translated = padded.gather(
        2,
        rows[:, None, :, None].expand(
            batch_size,
            channels,
            height,
            padded.shape[-1],
        ),
    )

    columns = torch.arange(width, device=images.device).unsqueeze(0)
    columns = columns + left.unsqueeze(1)
    translated = translated.gather(
        3,
        columns[:, None, None, :].expand(
            batch_size,
            channels,
            height,
            width,
        ),
    )
    return translated, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (original_logits + flipped_logits)

    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
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
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
    if step < warmup_steps:
        multiplier = 0.1 + 0.9 * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        multiplier = 0.02 + 0.98 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier
>>>>>>> REPLACE