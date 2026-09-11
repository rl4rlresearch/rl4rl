MECHANISM: Residual convolutional feature learning with flip-consistent ensembling

HYPOTHESIS: A 234,858-parameter residual BatchNorm CNN trained with twice as many optimizer updates will exceed the starting design’s 8,928 correct predictions; lightweight horizontal-flip training and evaluation ensembling will further improve Fashion-MNIST-style invariance.

INTENDED_EDIT: Replace the shallow CNN with a parameter-compliant residual network, use batch size 128, per-example horizontal flips, probability-space flip ensembling at evaluation, grouped AdamW decay, and a short-warmup cosine schedule.

EVIDENCE: The verified starting design reaches 89.28% accuracy with only 105,866 parameters and 392 updates, indicating unused capacity and optimization budget; the prior larger design was not verified, so this patch retains the promising capacity/update changes while using a simpler augmentation path.

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.skip = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.conv5 = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(96)
        self.pool = nn.MaxPool2d(2)
        self.spatial_pool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(96 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(images)), inplace=True)
        x = F.relu(x + self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)

        residual = self.skip(x)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(residual + self.bn4(self.conv4(x)), inplace=True)
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = self.spatial_pool(x).flatten(1)
        x = F.dropout(x, p=0.15, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=0.10, training=self.training)
        return self.fc2(x)

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
    decay = [parameter for parameter in model.parameters() if parameter.ndim > 1]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=6e-4,
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
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
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
    completed = step + 1
    warmup_steps = max(1, total_steps // 20)
    if completed <= warmup_steps:
        multiplier = 0.2 + 0.8 * completed / warmup_steps
    else:
        progress = min(
            1.0,
            (completed - warmup_steps)
            / max(total_steps - warmup_steps, 1),
        )
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
>>>>>>> REPLACE