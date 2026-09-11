MECHANISM: Parameter-efficient residual feature hierarchy with lightweight translation and reflection augmentation

HYPOTHESIS: A 241,130-parameter residual BatchNorm CNN trained for roughly twice the optimizer updates will exceed the verified baseline’s 8,928 correct predictions.

INTENDED_EDIT: Replace the shallow CNN with a four-block residual network, use batch size 128, add padded crops and per-image horizontal flips, and train with warmup-cosine AdamW.

EVIDENCE: The verified baseline achieves 89.28% accuracy using only 105,866 parameters and 392 updates, leaving substantial unused capacity and update budget; the unverified designs motivate retaining those core changes while omitting evaluation-time ensembling.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0


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
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.activation(outputs + residual)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 96, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(96, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(images))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    warmup_steps = max(1, int(0.04 * total_steps))
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-3 / warmup_steps,
        weight_decay=5e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    images = F.pad(images, (2, 2, 2, 2))
    offsets = torch.randint(0, 5, (2,))
    top, left = offsets.tolist()
    images = images[:, :, top : top + 28, left : left + 28]

    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.04 * total_steps))
    completed = step + 1

    if completed <= warmup_steps:
        learning_rate = peak_lr * completed / warmup_steps
    else:
        progress = (completed - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        progress = min(progress, 1.0)
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        learning_rate = peak_lr * multiplier

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE