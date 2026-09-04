MECHANISM: Deeper batch-normalized CNN with translation augmentation and warmup-cosine optimization

HYPOTHESIS: Expanding the under-capacity feature extractor while adding small label-preserving translations will exceed 8,928 correct predictions within the fixed exposure budget and 250,000-parameter ceiling.

INTENDED_EDIT: Use a 232k-parameter five-convolution network, batch size 128, per-image reflected translations and flips, lighter label smoothing, relaxed clipping, and a warmup-cosine AdamW schedule.

EVIDENCE: The starting two-convolution, 105,866-parameter model reached 89.28% accuracy, leaving substantial parameter capacity available for learning richer spatial features.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0

START_LR = 1.0e-3
MAX_LR = 3.0e-3
MIN_LR = 1.0e-4
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(96 * 3 * 3, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )
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
        lr=START_LR,
        weight_decay=2.0e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    starts = torch.randint(
        0,
        2 * padding + 1,
        (images.shape[0], 2),
        device=images.device,
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)[:, None, None]
    coordinates = torch.arange(28, device=images.device)
    rows = starts[:, 0, None, None] + coordinates[None, :, None]
    columns = starts[:, 1, None, None] + coordinates[None, None, :]
    images = padded[:, 0][batch_indices, rows, columns].unsqueeze(1)

    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
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
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.08
    if progress < warmup_fraction:
        learning_rate = START_LR + (MAX_LR - START_LR) * (
            progress / warmup_fraction
        )
    else:
        cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        learning_rate = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE