MECHANISM: Residual multiscale CNN with flip invariance

HYPOTHESIS: Using the available parameter headroom for batch-normalized residual features, while doubling optimizer updates and removing target smoothing, will exceed the baseline’s 8,928 correct predictions after the same 100,000 examples.

INTENDED_EDIT: Replace the shallow CNN with a 245,040-parameter residual depthwise model, use batch size 128, apply horizontal-flip augmentation and matching test-time averaging, and train with unsmoothed cross-entropy.

EVIDENCE: The baseline reaches 8,928 correct with only 105,866 parameters, one convolution per spatial scale, 392 optimizer steps, and label smoothing; substantial capacity and update-count headroom remains below the 250,000-parameter ceiling.

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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))

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
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    del step, total_steps
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE