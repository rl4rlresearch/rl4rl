MECHANISM: Compute-aware low-resolution residual refinement

HYPOTHESIS: A 230,442-parameter CNN that concentrates added capacity at 7×7 resolution, paired with twice as many optimizer updates and mild translation/flip augmentation, will exceed the verified baseline’s 8,928 correct predictions without the verification risk of the prior deeper high-resolution designs.

INTENDED_EDIT: Add a compact 7×7 residual block and wider classifier, use batch size 128, reflected translations and horizontal flips, reduce label smoothing, relax clipping, and apply warmup-cosine AdamW.

EVIDENCE: The 105,866-parameter baseline achieved 89.28% accuracy, demonstrating capacity headroom; two larger multi-convolution attempts could not be verified, motivating a simpler architecture that adds most computation only after spatial downsampling.

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
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
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
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
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
    offsets = torch.randint(0, 5, (2,))
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    images = padded[:, :, offset_y : offset_y + 28, offset_x : offset_x + 28]
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
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
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
>>>>>>> REPLACE