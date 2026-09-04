MECHANISM: Flip-ensemble inference over the EMA-trained convolutional network

HYPOTHESIS: Averaging predictions for each image and its horizontal mirror will exceed the reference’s 9,099 correct predictions because the qualified model was trained with random horizontal flips, while preserving its parameter count and exposure budget.

INTENDED_EDIT: Adopt the qualified six-layer CNN and training procedure, then add evaluation-time probability averaging between original and mirrored inputs.

EVIDENCE: Reference Design 1 achieved 9,099/10,000 correct with 245,818 parameters; its 50% horizontal-flip augmentation directly supports a low-variance mirrored-view ensemble.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0
BASE_LR = 3.0e-3
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

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(images.flip(-1))
        log_probs = F.log_softmax(logits, dim=1)
        flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
        return torch.logaddexp(log_probs, flipped_log_probs) - math.log(2.0)
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