MECHANISM: Parallel local/dilated residual features with spatial-pyramid classification

HYPOTHESIS: Reallocating the location-sensitive flattened head into a 96-channel local/dilated stage and pooled coarse-shape/saliency head will exceed 9,175 correct predictions by learning more discriminative garment-scale structure.

INTENDED_EDIT: Add a parallel standard/dilated residual block, widen the final representation from 64 to 96 channels, and replace the 7×7 flattening bottleneck with 2×2 average plus global-max pooling.

EVIDENCE: Augmentation and inference refinements plateaued near 9,175 correct, while the current model spends 94,420 parameters compressing exact 7×7 positions into only 30 features. This patch instead uses 247,142 parameters to learn local and larger-context features while retaining coarse layout and translation-tolerant saliency.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 2
        self.local = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.fuse_bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = torch.cat(
            (self.local(inputs), self.context(inputs)),
            dim=1,
        )
        outputs = self.fuse_bn(self.fuse(outputs))
        return F.gelu(outputs + residual)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            MultiScaleResidualBlock(64, 96),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 5, 52),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(52, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.features(images)
        coarse_shape = F.adaptive_avg_pool2d(
            feature_maps, output_size=(2, 2)
        ).flatten(1)
        salient_features = F.adaptive_max_pool2d(
            feature_maps, output_size=1
        ).flatten(1)
        pooled = torch.cat((coarse_shape, salient_features), dim=1)
        return self.classifier(pooled)
>>>>>>> REPLACE