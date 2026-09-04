MECHANISM: Per-image integer translation regularization on the verified update-dense CNN

HYPOTHESIS: The verified 243,178-parameter design with mild ±2-pixel translation augmentation, but no horizontal flipping, will exceed 9,258 correct validation predictions.

INTENDED_EDIT: Adopt Reference Design 1’s architecture, batch size, optimizer, loss, and schedule, then add independently sampled integer translations using replicated-border crops.

EVIDENCE: Reference Design 1 achieved 9,258 correct without augmentation, while horizontal flips reduced that to 9,230; this motivates preserving the verified core and testing only a label-preserving invariance better suited to centered 28×28 images.

<<<<<<< SEARCH
BATCH_SIZE = 256
=======
BATCH_SIZE = 128
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


def training_loss(
=======
    del step, total_steps
    padding = 2
    padded = F.pad(
        images, (padding, padding, padding, padding), mode="replicate"
    )
    batch, channels, height, width = images.shape
    padded_width = width + 2 * padding
    offset_y = torch.randint(
        0, 2 * padding + 1, (batch, 1, 1), device=images.device
    )
    offset_x = torch.randint(
        0, 2 * padding + 1, (batch, 1, 1), device=images.device
    )
    rows = torch.arange(height, device=images.device).view(1, height, 1)
    columns = torch.arange(width, device=images.device).view(1, 1, width)
    indices = (rows + offset_y) * padded_width + columns + offset_x
    indices = indices.reshape(batch, 1, height * width).expand(
        -1, channels, -1
    )
    images = torch.gather(padded.flatten(2), 2, indices).reshape_as(images)
    return images, labels


def training_loss(
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