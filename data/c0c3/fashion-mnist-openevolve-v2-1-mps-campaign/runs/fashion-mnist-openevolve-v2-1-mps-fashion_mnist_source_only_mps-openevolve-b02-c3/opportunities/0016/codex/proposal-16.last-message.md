MECHANISM: Boundary-consistent whole-map residual refinement

HYPOTHESIS: Replicate-padding the best 7×7 depthwise refinement block will exceed 9,270 correct predictions by preventing its whole-map kernel from mixing late features with artificial zero borders.

INTENDED_EDIT: Adopt the qualified 7×7 residual spatial-head design and matched axial training/TTA recipe, changing the refinement block’s depthwise convolution to replicate padding.

EVIDENCE: The single 7×7 block achieved the best result at 9,270 correct, extending the improvement from 3×3 to 5×5, while its large kernel makes boundary padding especially influential; the successful augmentation recipe already uses replicate-padded translations.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0

PEAK_LR = 3.0e-3
MIN_LR_RATIO = 0.05
WARMUP_FRACTION = 0.08


class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(images + refined)
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
            SpatialRefinement(64),
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
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        logits = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        for crop, weight in zip(crops, crop_weights):
            logits = logits + weight * self._forward_once(crop)
            logits = logits + weight * self._forward_once(crop.flip(-1))
        return logits / (2.0 * sum(crop_weights))
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

    padding = 1
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
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