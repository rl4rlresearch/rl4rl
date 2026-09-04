MECHANISM: Flip-invariant update-dense CNN with evaluation-time logit averaging

HYPOTHESIS: Preserving the verified 243,178-parameter design while adding label-preserving horizontal-flip training and two-view inference will exceed 9,258 correct predictions on the fixed validation set.

INTENDED_EDIT: Adopt the qualified batch-normalized CNN and optimizer schedule, randomly flip individual training images, and average original and mirrored logits during evaluation.

EVIDENCE: The qualified reference achieved 9,258 correct and 0.21875 cross-entropy without augmentation; retaining that verified core isolates horizontal-flip invariance as the next informative improvement, whereas earlier augmented proposals were not successfully verified.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0
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
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1, groups=128, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 10),
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
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if parameter.ndim == 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.5e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    flip_mask = torch.rand(
        (images.shape[0], 1, 1, 1), device=images.device
    ) < 0.5
    images = torch.where(flip_mask, torch.flip(images, dims=(-1,)), images)
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
    peak_lr = 2.5e-3
    minimum_lr = 1.0e-4
    completed = min(step + 1, max(total_steps, 1))
    warmup_steps = max(1, int(0.05 * total_steps))
    if completed <= warmup_steps:
        fraction = completed / warmup_steps
        learning_rate = peak_lr * (0.1 + 0.9 * fraction)
    else:
        fraction = (completed - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        learning_rate = minimum_lr + (peak_lr - minimum_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE