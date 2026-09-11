MECHANISM: Residual multiscale CNN with light geometric augmentation and flip-ensemble inference

HYPOTHESIS: Increasing feature capacity while remaining below 250,000 parameters, doubling optimizer updates, and training for flip/translation invariance will exceed the current 8,928 correct predictions and reduce the 0.3359 validation cross-entropy.

INTENDED_EDIT: Replace the small two-layer CNN with a batch-normalized residual CNN, use batch size 128, add horizontal-flip and translation augmentation, average flipped logits during validation, and apply warmup-cosine AdamW optimization.

EVIDENCE: The current 105,866-parameter model reaches 89.28% accuracy after only 392 optimizer steps, leaving substantial parameter and update capacity available under the fixed exposure and 250,000-parameter limits.

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
class ResidualRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.block(inputs))


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
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualRefinement(96),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 104),
            nn.LayerNorm(104),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(104, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=5e-4)
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, torch.flip(images, dims=(-1,)), images)

    offsets = torch.randint(-2, 3, (2,))
    images = torch.roll(
        images,
        shifts=(int(offsets[0]), int(offsets[1])),
        dims=(-2, -1),
    )
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.08
    maximum_lr = 3.0e-3
    minimum_lr = 1.0e-4
    if progress < warmup_fraction:
        warmup_progress = progress / warmup_fraction
        learning_rate = maximum_lr * (0.25 + 0.75 * warmup_progress)
    else:
        cosine_progress = (
            (progress - warmup_fraction) / (1.0 - warmup_fraction)
        )
        multiplier = 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
        learning_rate = minimum_lr + (
            maximum_lr - minimum_lr
        ) * multiplier

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE