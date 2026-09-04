MECHANISM: Per-image integer-translation augmentation with paired flip supervision

HYPOTHESIS: Restoring the 9,320-correct shared dual-statistic attention model and adding mild per-image translations will exceed 9,320 correct predictions by improving spatial robustness without changing parameter count or the successful paired-view objective.

INTENDED_EDIT: Restore Reference Design 1, then apply independent replicate-padded translations of up to two pixels before its random flip and paired-orientation training.

EVIDENCE: Shared average-plus-maximum attention with paired supervision achieved the best result at 9,320 correct. Translations previously appeared only in a confounded 9,024-correct experiment that also changed batch size and classifier capacity, so isolating translation augmentation on the best regimen is informative.

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
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(32)
        self.stem_refine_conv1 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm1 = nn.BatchNorm2d(32)
        self.stem_refine_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm2 = nn.BatchNorm2d(32)

        self.block_conv1 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, bias=False
        )
        self.block_norm1 = nn.BatchNorm2d(64)
        self.block_conv2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.block_norm2 = nn.BatchNorm2d(64)
        self.projection = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)

        self.deep_conv1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.deep_norm1 = nn.BatchNorm2d(32)
        self.deep_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.deep_norm2 = nn.BatchNorm2d(32)
        self.deep_conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.deep_norm3 = nn.BatchNorm2d(64)

        self.post_deep_conv1 = nn.Conv2d(64, 24, kernel_size=1, bias=False)
        self.post_deep_norm1 = nn.BatchNorm2d(24)
        self.post_deep_conv2 = nn.Conv2d(
            24, 24, kernel_size=3, padding=1, bias=False
        )
        self.post_deep_norm2 = nn.BatchNorm2d(24)
        self.post_deep_conv3 = nn.Conv2d(24, 64, kernel_size=1, bias=False)
        self.post_deep_norm3 = nn.BatchNorm2d(64)

        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.stem_norm(self.stem_conv(images)))
        residual = features
        features = F.gelu(
            self.stem_refine_norm1(self.stem_refine_conv1(features))
        )
        features = self.stem_refine_norm2(self.stem_refine_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = features
        features = F.gelu(self.deep_norm1(self.deep_conv1(features)))
        features = F.gelu(self.deep_norm2(self.deep_conv2(features)))
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)

        residual = features
        features = F.gelu(
            self.post_deep_norm1(self.post_deep_conv1(features))
        )
        features = F.gelu(
            self.post_deep_norm2(self.post_deep_conv2(features))
        )
        features = self.post_deep_norm3(self.post_deep_conv3(features))
        features = F.gelu(features + residual)

        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
        features = features * (2.0 * torch.sigmoid(channel_gate))

        spatial_summary = torch.cat(
            (
                features.mean(dim=1, keepdim=True),
                features.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = self.spatial_attention(spatial_summary)
        features = features * (2.0 * torch.sigmoid(spatial_gate))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
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
        lr=2.5e-3,
        weight_decay=5e-4,
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
    height, width = images.shape[-2:]
    images = F.pad(images, (2, 2, 2, 2), mode="replicate")
    offsets = torch.randint(
        0,
        5,
        (images.shape[0], 2),
        device=images.device,
    )

    row_indices = offsets[:, :1] + torch.arange(height, device=images.device)
    images = images.gather(
        2,
        row_indices[:, None, :, None].expand(
            -1, images.shape[1], -1, images.shape[3]
        ),
    )
    column_indices = offsets[:, 1:] + torch.arange(width, device=images.device)
    images = images.gather(
        3,
        column_indices[:, None, None, :].expand(
            -1, images.shape[1], images.shape[2], -1
        ),
    )

    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
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
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
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
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE