MECHANISM: Deeper residual feature learning with parameter-efficient head and translated crops

HYPOTHESIS: A 232,778-parameter two-block residual CNN trained with 64-image batches and mild translation augmentation will exceed the reference design’s 9,141 correct predictions by shifting capacity from its large dense layer into spatial feature extraction and providing roughly twice as many optimizer updates.

INTENDED_EDIT: Add a second residual block, reduce the dense head width, use batch size 64, introduce random two-pixel translations, and retune AdamW’s learning rate for the smaller batch.

EVIDENCE: The 209,146-parameter residual reference reached 91.41% after 782 updates, substantially outperforming the shallow model; its 150,528-weight first dense layer leaves room to deepen convolutional processing while remaining below the parameter ceiling.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 64
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

        self.refine_conv1 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.refine_norm1 = nn.BatchNorm2d(64)
        self.refine_conv2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.refine_norm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(F.gelu(self.stem_norm(self.stem_conv(images))))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = features
        features = F.gelu(self.refine_norm1(self.refine_conv1(features)))
        features = self.refine_norm2(self.refine_conv2(features))
        features = F.gelu(features + residual)
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
        lr=1.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )

    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding))
    batch = images.shape[0]
    offset_y = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    offset_x = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    row_index = (
        torch.arange(images.shape[-2], device=images.device)[None, :, None]
        + offset_y[:, None, None]
    )
    column_index = (
        torch.arange(images.shape[-1], device=images.device)[None, None, :]
        + offset_x[:, None, None]
    )
    images = padded[:, 0][batch_index, row_index, column_index].unsqueeze(1)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 1.5e-3 * multiplier
>>>>>>> REPLACE