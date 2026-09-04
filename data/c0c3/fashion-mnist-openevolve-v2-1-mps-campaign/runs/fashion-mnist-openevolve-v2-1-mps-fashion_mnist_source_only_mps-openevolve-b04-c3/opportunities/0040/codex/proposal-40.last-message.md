MECHANISM: Content-adaptive multi-part attention pooling

HYPOTHESIS: Replacing the fixed-position 4,704-to-38 bottleneck with four learned spatial part summaries plus a global summary will exceed 9,240 correct predictions by preserving channel-rich local evidence while retaining the verified backbone’s runtime profile.

INTENDED_EDIT: Restore the verified depthwise residual backbone, flip training and inference, and 12.5%-floor schedule, but compute predictions through initialized quadrant-attention pools and a 372-unit head; the model has 249,774 learned parameters.

EVIDENCE: The fixed-flatten design reached 9,240 correct, while widening its narrow classifier fell to 9,223 and added spatial convolutions timed out; reallocating the same head compute to content-dependent part aggregation cleanly challenges the shared fixed-location bottleneck without increasing convolutional work.

<<<<<<< SEARCH
BATCH_SIZE = 256
=======
BATCH_SIZE = 64
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
class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.depthwise(inputs)))
        hidden = self.norm2(self.pointwise(hidden))
        return F.gelu(inputs + hidden)


class PartAttentionPool(nn.Module):
    def __init__(self, channels: int, parts: int = 4) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, parts, kernel_size=1)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

        position = torch.zeros(1, parts, 7, 7)
        position[:, 0, :4, :4] = 1.0
        position[:, 1, :4, 3:] = 1.0
        position[:, 2, 3:, :4] = 1.0
        position[:, 3, 3:, 3:] = 1.0
        self.position_bias = nn.Parameter(position)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scores = self.attention(features) + self.position_bias
        weights = F.softmax(scores.flatten(2), dim=-1)
        values = features.flatten(2).transpose(1, 2)
        parts = torch.bmm(weights, values).flatten(1)
        global_summary = features.mean(dim=(2, 3))
        return torch.cat((global_summary, parts), dim=1)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
        )
        self.pool = PartAttentionPool(96)
        self.classifier = nn.Sequential(
            nn.Linear(96 * 5, 372),
            nn.LayerNorm(372),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(372, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.stem(images)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE