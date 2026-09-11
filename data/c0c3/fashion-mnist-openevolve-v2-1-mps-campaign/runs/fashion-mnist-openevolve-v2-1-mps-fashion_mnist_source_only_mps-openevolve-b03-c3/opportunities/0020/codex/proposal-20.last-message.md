MECHANISM: Class-conditional multiscale evidence pooling

HYPOTHESIS: Replacing the flattened single-scale classifier with dilated multiscale residual features, class-specific spatial attention, and coarse average/max shape pooling will exceed 9,237 correct predictions by learning translation-tolerant local evidence while retaining coarse spatial structure.

INTENDED_EDIT: Restore the strongest weighted-translation, flip-complete training recipe, but replace its load-bearing sequential 7×7 flattening assumption with a 242,910-parameter multiscale encoder and dual local/global prediction head.

EVIDENCE: Differential-basis encoding raised the flip-complete design from 9,186 to 9,224 correct, but subsequent loss-schedule tuning plateaued at 9,237 and another conventional residual refinement reached only 9,039; this motivates reallocating capacity from absolute-position flattening to a genuinely different class-conditional evidence computation.

<<<<<<< SEARCH
BATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 192
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
class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.context = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = F.gelu(self.local(features))
        context = F.gelu(self.context(features))
        update = self.norm(self.mix(torch.cat((local, context), dim=1)))
        return F.gelu(features + update)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.high_resolution = MultiScaleBlock(48)
        self.transition = nn.Sequential(
            nn.Conv2d(48, 112, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(112),
            nn.GELU(),
        )
        self.low_resolution = nn.Sequential(
            MultiScaleBlock(112),
            MultiScaleBlock(112),
            MultiScaleBlock(112),
        )

        self.local_evidence = nn.Conv2d(112, 10, kernel_size=1)
        self.local_attention = nn.Conv2d(112, 10, kernel_size=1)
        self.coarse_classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(112 * 2 * 2 * 2),
            nn.Linear(112 * 2 * 2 * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 10),
        )

    @staticmethod
    def _image_basis(images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
        local_contrast = images - local_mean

        gradient_x = 0.125 * (
            padded[:, :, :-2, 2:]
            + 2.0 * padded[:, :, 1:-1, 2:]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, 1:-1, :-2]
            - padded[:, :, 2:, :-2]
        )
        gradient_y = 0.125 * (
            padded[:, :, 2:, :-2]
            + 2.0 * padded[:, :, 2:, 1:-1]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, :-2, 1:-1]
            - padded[:, :, :-2, 2:]
        )
        edge_energy = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1.0e-6
        )
        return torch.cat(
            (
                images,
                local_contrast,
                gradient_x,
                gradient_y,
                edge_energy,
            ),
            dim=1,
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(self._image_basis(images))
        features = self.high_resolution(features)
        features = F.max_pool2d(features, kernel_size=2)
        features = self.transition(features)
        features = self.low_resolution(features)

        evidence = self.local_evidence(features).flatten(2)
        attention = self.local_attention(features).flatten(2).softmax(dim=-1)
        local_logits = (evidence * attention).sum(dim=-1)

        coarse_features = torch.cat(
            (
                F.adaptive_avg_pool2d(features, output_size=2),
                F.adaptive_max_pool2d(features, output_size=2),
            ),
            dim=1,
        )
        return local_logits + self.coarse_classifier(coarse_features)

    def _flip_average(self, images: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images)
            + self._forward_once(images.flip(-1))
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        logits = 0.36 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
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
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    translation_weights = images.new_tensor(
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
    )
    flat_offsets = torch.multinomial(
        translation_weights, images.size(0), replacement=True
    )
    offsets = torch.stack(
        (flat_offsets.div(5, rounding_mode="floor"), flat_offsets.remainder(5)),
        dim=1,
    )

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    row_index = rows[:, None, :, None].expand(
        -1, images.size(1), -1, padded.size(3)
    )
    images = padded.gather(2, row_index)

    columns = offsets[:, 1, None] + torch.arange(width, device=images.device)
    column_index = columns[:, None, None, :].expand(
        -1, images.size(1), height, -1
    )
    images = images.gather(3, column_index)

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
    progress = min(step / max(total_steps, 1), 1.0)
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
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
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE