MECHANISM: Capacity-scaled convolutional ensemble with geometric invariance

HYPOTHESIS: A wider two-convolution-per-stage network trained with random translations and horizontal flips, then evaluated by averaging original and mirrored logits, will exceed the current 8,928 correct predictions while remaining below 250,000 parameters.

INTENDED_EDIT: Use batch size 128, a 241,426-parameter batch-normalized CNN, vectorized crop-and-flip augmentation, a short-warmup cosine AdamW schedule, lighter label smoothing, and horizontal-flip test-time ensembling.

EVIDENCE: The starting 105,866-parameter two-convolution model reaches 89.28% accuracy after 100,000 examples without augmentation, leaving substantial parameter headroom and no explicit geometric invariance.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0

PEAK_LR = 3.0e-3
MIN_LR_RATIO = 0.05
WARMUP_FRACTION = 0.08
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
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
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
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
    batch = images.shape[0]

    flip_mask = torch.rand(
        (batch, 1, 1, 1), device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)

    padding = 2
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    offsets_y = torch.randint(
        0, 2 * padding + 1, (batch, 1), device=images.device
    )
    offsets_x = torch.randint(
        0, 2 * padding + 1, (batch, 1), device=images.device
    )
    coordinates = torch.arange(28, device=images.device).unsqueeze(0)
    rows = coordinates + offsets_y
    columns = coordinates + offsets_x
    batch_indices = torch.arange(batch, device=images.device)[:, None, None]
    images = padded[:, 0][
        batch_indices,
        rows[:, :, None],
        columns[:, None, :],
    ].unsqueeze(1)

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
    progress = min(step / max(total_steps, 1), 1.0)
    if progress < WARMUP_FRACTION:
        multiplier = 0.2 + 0.8 * progress / WARMUP_FRACTION
    else:
        cosine_progress = (
            progress - WARMUP_FRACTION
        ) / (1.0 - WARMUP_FRACTION)
        multiplier = MIN_LR_RATIO + (
            1.0 - MIN_LR_RATIO
        ) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
>>>>>>> REPLACE