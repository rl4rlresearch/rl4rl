MECHANISM: Lossless space-to-depth residual pyramid classifier

HYPOTHESIS: Replacing lossy max-pooling and the positional dense head with lossless pixel rearrangement, residual feature learning, and coarse spatial-pyramid pooling will achieve at least 9,287 validation-correct predictions within the time limit.

INTENDED_EDIT: Move capacity from the flattening MLP into two residual stages, preserve pixels during downsampling with PixelUnshuffle, classify pooled global and quadrant features, and batch the existing TTA views.

EVIDENCE: The 9,286-correct baseline assumes repeated max-pooling is sufficient, while the multi-resolution alternative timed out after adding a second branch. This replacement tests preserved spatial evidence without adding a parallel computational path and uses 233,194 parameters with fewer convolutional operations than the baseline.

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
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
=======
class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return F.gelu(hidden + inputs)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(4, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock(48)
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(48 * 4, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.stage2 = ResidualBlock(96)
        self.dropout = nn.Dropout(0.10)
        self.classifier = nn.Linear(96 * 5, 10)

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stage1(self.stem(images))
        features = self.stage2(self.downsample(features))
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        quadrant_features = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        pooled = torch.cat((global_features, quadrant_features), dim=1)
        return self.classifier(self.dropout(pooled))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 0.912
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._predict(images)
        if self.training:
            return logits

        batch = images.shape[0]
        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        translated = torch.cat(views[1:], dim=0)
        translated_logits = self._predict(translated).reshape(4, batch, 10)
        flipped = torch.cat(tuple(view.flip(-1) for view in views), dim=0)
        flipped_logits = self._predict(flipped).reshape(5, batch, 10)

        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        for view_logits in translated_logits:
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view_logits in enumerate(flipped_logits):
            view_log_probs = F.log_softmax(view_logits, dim=1)
            if view_index == 0:
                view_log_probs = view_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, view_log_probs)
        return (ensemble - math.log(12.0)) / 0.912
>>>>>>> REPLACE