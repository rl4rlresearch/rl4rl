MECHANISM: Convolution-heavy residual hierarchy with spatially pooled classification

HYPOTHESIS: Reallocating the reference model’s large dense layer into residual spatial convolutions, while increasing optimizer updates with batch size 128, will exceed 9,009 correct validation predictions within the same 100,000-example budget.

INTENDED_EDIT: Build a 247,794-parameter three-stage residual CNN, retain proven translation/flip augmentation and flip-averaged evaluation, and train it with decoupled AdamW, mild label smoothing, and cosine decay.

EVIDENCE: The 241,682-parameter reference reached 9,009 correct, but 188,160 weights were concentrated in its first dense layer; using that capacity for hierarchical convolution should learn stronger shape features while preserving its validated augmentation and evaluation strategy.

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
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.transition3 = nn.Sequential(
            nn.Conv2d(64, 110, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(110),
            nn.GELU(),
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(110, 110, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(110),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Dropout(0.05),
            nn.Linear(110 * 3 * 3, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.refine1(features))
        features = self.pool(features)
        features = self.transition2(features)
        features = F.gelu(features + self.refine2(features))
        features = self.pool(features)
        features = self.transition3(features)
        features = F.gelu(features + self.refine3(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            logits = 0.5 * (
                logits + self._forward_once(images.flip(-1))
            )
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
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
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
    height, width = images.shape[-2:]
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    offsets = torch.randint(0, 5, (images.size(0), 2), device=images.device)

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    row_index = rows[:, None, :, None].expand(
        -1, images.size(1), -1, padded.size(3)
    )
    images = padded.gather(2, row_index)

    columns = offsets[:, 1, None] + torch.arange(width, device=images.device)
    column_index = columns[:, None, None, :].expand(
        -1, images.size(1), height, -1
    )
    images = images.gather(3, column_index)

    flip_mask = torch.rand(images.size(0), device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None], images.flip(-1), images
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
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE