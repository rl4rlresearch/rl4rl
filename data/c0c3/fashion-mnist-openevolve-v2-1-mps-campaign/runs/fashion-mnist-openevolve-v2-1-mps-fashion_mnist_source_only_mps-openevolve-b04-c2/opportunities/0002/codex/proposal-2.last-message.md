MECHANISM: Update-dense batch-normalized CNN with translation/flip augmentation and flip-ensemble inference

HYPOTHESIS: A 229,098-parameter six-layer CNN trained with twice as many optimizer updates will exceed 8,928 validation-correct predictions; light geometric augmentation and averaged flipped logits should also improve the 0.3359 cross-entropy.

INTENDED_EDIT: Use batch size 128, a deeper batch-normalized CNN, random per-image flips plus batchwise translations, evaluation-time flip averaging, lightly smoothed cross-entropy, and warmup-cosine AdamW.

EVIDENCE: The verified 105,866-parameter baseline reaches 8,928 correct with only 392 optimizer steps, leaving substantial capacity and update-frequency headroom; the previous more ambitious residual proposal was unverified, motivating a simpler implementation using only standard PyTorch operations.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
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
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 2 * 2, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

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
        lr=5.0e-4,
        betas=(0.9, 0.99),
        weight_decay=2.0e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    height, width = images.shape[-2:]
    padding = 2
    offset_y = int(
        torch.randint(0, 2 * padding + 1, (), device=images.device).item()
    )
    offset_x = int(
        torch.randint(0, 2 * padding + 1, (), device=images.device).item()
    )
    images = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    images = images[
        ..., offset_y : offset_y + height, offset_x : offset_x + width
    ]
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
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
    peak_lr = 2.5e-3
    final_lr = 7.5e-5
    warmup_steps = max(1, int(0.05 * total_steps))
    next_step = min(step + 1, total_steps)

    if next_step < warmup_steps:
        lr = peak_lr * (0.2 + 0.8 * next_step / warmup_steps)
    else:
        progress = (next_step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = final_lr + (peak_lr - final_lr) * cosine

    for group in optimizer.param_groups:
        group["lr"] = lr
>>>>>>> REPLACE