MECHANISM: High-capacity batch-normalized CNN with lightweight geometric augmentation

HYPOTHESIS: Expanding the 105,866-parameter two-layer CNN to a roughly 246k-parameter six-layer CNN, while doubling optimizer updates with batch size 128, will exceed 8,928 correct predictions by learning more discriminative features within the fixed 100,000-example exposure.

INTENDED_EDIT: Replace the baseline with a six-convolution batch-normalized network, random per-image translation and horizontal flipping, AdamW with warmup-cosine scheduling, mild label smoothing, and end-of-training exponential weight averaging.

EVIDENCE: The starting design reaches 8,928/10,000 using only 105,866 of the allowed 250,000 parameters and 392 optimizer steps, leaving substantial capacity and optimization-step budget available.

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
GRAD_CLIP_NORM = 2.0
BASE_LR = 3.0e-3


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
        for index, (in_channels, out_channels) in enumerate(channels):
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        weight_decay=3e-4,
    )
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
    optimizer._ema_started = False
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    batch_indices = torch.arange(images.shape[0], device=images.device)
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    offsets_y = torch.randint(0, 5, (images.shape[0],), device=images.device)
    offsets_x = torch.randint(0, 5, (images.shape[0],), device=images.device)
    images = crops[batch_indices, :, offsets_y, offsets_x]
    flip_mask = torch.rand(
        (images.shape[0], 1, 1, 1), device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    warmup_steps = max(1, total_steps // 20)
    if step < warmup_steps:
        multiplier = 0.2 + 0.8 * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        multiplier = 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = BASE_LR * multiplier

    with torch.no_grad():
        if not optimizer._ema_started:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.copy_(tensor)
            optimizer._ema_started = True
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.01)

        if step >= total_steps:
            for tensor, average in zip(
                optimizer._ema_tensors, optimizer._ema_values
            ):
                tensor.copy_(average)
>>>>>>> REPLACE