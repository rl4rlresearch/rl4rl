MECHANISM: Batch-normalized wider CNN with per-image translation/flip augmentation and flip-ensemble inference

HYPOTHESIS: Increasing capacity from 105,866 to about 216,000 parameters while doubling optimizer updates with batch size 128 will raise validation_correct above the 8,928 baseline; normalization, mild augmentation, and flip averaging will improve fixed-exposure generalization.

INTENDED_EDIT: Replace the shallow CNN with a two-stage normalized CNN, use batch size 128, per-image crop/flip augmentation, faster-adapting AdamW, warmup-cosine scheduling, and probability-averaged horizontal-flip inference.

EVIDENCE: The starting design reaches 8,928 correct after only 392 optimizer steps with 105,866 parameters, leaving substantial room under the 250,000-parameter ceiling and motivating more capacity and optimization updates.

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
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(images.flip(-1))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)
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
        lr=6.0e-4,
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
    batch = images.shape[0]

    flip_mask = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)

    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    height, width = images.shape[-2:]
    offsets_y = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    offsets_x = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    rows = offsets_y[:, None, None] + torch.arange(
        height, device=images.device
    )[None, :, None]
    columns = offsets_x[:, None, None] + torch.arange(
        width, device=images.device
    )[None, None, :]
    images = padded[:, 0][batch_index, rows, columns].unsqueeze(1).contiguous()
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    warmup_fraction = 0.05
    start_lr = 6.0e-4
    peak_lr = 3.0e-3
    end_lr = 1.0e-4
    if progress < warmup_fraction:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
>>>>>>> REPLACE