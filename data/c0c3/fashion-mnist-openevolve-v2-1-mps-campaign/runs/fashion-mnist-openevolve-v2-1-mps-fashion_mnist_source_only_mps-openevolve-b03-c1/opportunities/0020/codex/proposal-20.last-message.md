MECHANISM: Content-adaptive spatial evidence pooling with residual feature extraction

HYPOTHESIS: Replacing the fixed 3×3 flattening head with learned spatial attention plus peak pooling, while reallocating parameters to wider residual features, will exceed 9,133 correct predictions by making class evidence less dependent on absolute crop position.

INTENDED_EDIT: Widen the convolutional hierarchy, preserve a 4×4 final feature map, add a depthwise residual refinement block, and classify from attended and maximum spatial summaries while remaining under the 250,000-parameter ceiling.

EVIDENCE: Differential input features raised accuracy from 9,091 to 9,122, showing that better learned representations remain valuable, while translation augmentation and multi-view inference produced repeated gains; this challenges the old assumption that a large fixed-position flattening head is the best use of parameters and motivates content-dependent spatial aggregation.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class DepthwiseResidualBlock(nn.Module):
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
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(images)
        residual = F.gelu(self.depthwise_norm(residual))
        residual = self.pointwise_norm(self.pointwise(residual))
        return F.gelu(images + residual)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(96, 136, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(136),
            nn.GELU(),
            DepthwiseResidualBlock(136),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(136, 34, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(34, 1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(136 * 2, 112),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(112, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        represented = torch.cat((images, details), dim=1)
        return self.classifier(self.features(represented))
=======
        represented = torch.cat((images, details), dim=1)
        features = self.features(represented)
        feature_tokens = features.flatten(2)
        attention = torch.softmax(
            self.spatial_attention(features).flatten(2),
            dim=-1,
        )
        attended = (feature_tokens * attention).sum(dim=-1)
        peak = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((attended, peak), dim=1))
>>>>>>> REPLACE