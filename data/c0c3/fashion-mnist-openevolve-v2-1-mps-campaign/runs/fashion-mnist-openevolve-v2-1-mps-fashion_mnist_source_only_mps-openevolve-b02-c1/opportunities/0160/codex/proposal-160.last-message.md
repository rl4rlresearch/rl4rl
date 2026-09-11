MECHANISM: Channel-gated residual multiscale spatial-pyramid classification

HYPOTHESIS: Reallocating parameters from the terminal flattening head into efficient residual feature refinement and jointly classifying 2×2 summaries from three resolutions will exceed 9,311 correct predictions by combining local texture evidence with progressively larger-scale shape evidence.

INTENDED_EDIT: Replace the single-scale sequential CNN with depthwise residual channel-gated stages, learned max/average downsampling, and a fused multiscale spatial-pyramid head while retaining the established optimizer, augmentation, and TTA calibration.

EVIDENCE: The current design plateaued at 9,311 correct while temperature changes affected only cross-entropy, showing that further progress requires different learned predictions. Its load-bearing assumption is that only the terminal 3×3 map merits classification capacity; the strong translation-aware ensemble instead suggests useful evidence exists across spatial scales. Unlike the timed-out attention and bilinear designs, this alternative uses inexpensive convolutional residual refinement and fixed-size pooling.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9340)
=======
class ChannelGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(2, 3))
        scale = torch.sigmoid(self.expand(F.gelu(self.reduce(pooled))))
        return features * scale[:, :, None, None]


class SeparableResidual(nn.Module):
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
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.gate = ChannelGate(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise_norm(self.depthwise(features))
        residual = F.gelu(residual)
        residual = self.pointwise_norm(self.pointwise(residual))
        residual = self.gate(residual)
        return F.gelu(features + residual)


class MixedPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mix = torch.sigmoid(self.mix)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return mix * maximum + (1.0 - mix) * average


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SeparableResidual(32),
        )
        self.down1 = MixedPool(32)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SeparableResidual(64),
        )
        self.down2 = MixedPool(64)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            SeparableResidual(96),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * (32 + 64 + 96), 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        level1 = self.stage1(self.stem(images))
        level2 = self.stage2(self.down1(level1))
        level3 = self.stage3(self.down2(level2))
        pyramid = torch.cat(
            (
                F.adaptive_avg_pool2d(level1, (2, 2)).flatten(1),
                F.adaptive_avg_pool2d(level2, (2, 2)).flatten(1),
                F.adaptive_avg_pool2d(level3, (2, 2)).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(pyramid)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._predict(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.0
        for view in views[1:]:
            ensemble = ensemble + self._predict(view)
        for view_index, view in enumerate(views):
            flipped_logits = self._predict(view.flip(-1))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9340)
>>>>>>> REPLACE