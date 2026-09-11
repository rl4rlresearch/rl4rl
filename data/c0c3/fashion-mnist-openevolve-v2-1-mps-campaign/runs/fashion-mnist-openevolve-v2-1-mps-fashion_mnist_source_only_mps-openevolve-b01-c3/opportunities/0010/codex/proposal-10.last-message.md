MECHANISM: Parallel multi-scale context fusion with global statistical pooling

HYPOTHESIS: Replacing the sequential late CNN and location-specific flattening head with parallel standard/dilated receptive fields plus global mean/max statistics will exceed 9,112 correct predictions by learning complementary texture and silhouette representations.

INTENDED_EDIT: Replace the classifier architecture with a 240,106-parameter residual multi-scale CNN while retaining the proven augmentation, optimization, and hierarchical test-time ensemble.

EVIDENCE: Successive spatial-refinement changes improved correctness from 9,035 to 9,086, whereas extensive ensemble refinements plateaued at 9,112. This suggests representation is now the limiting factor. The old design assumes sequential 3×3 features and a flattened 7×7 head are sufficient; this patch instead learns parallel local/contextual features and translation-tolerant global statistics.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.late1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.late2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 44),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(44, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mid(self.stem(images))
        features = F.gelu(features + self.late2(self.late1(features)))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._predict(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = []
        for row_offset, col_offset in (
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
        ):
            view = padded[
                :,
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            views.append(view)
            views.append(view.flip(-1))

        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
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
                channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat(
            (self.local(features), self.context(features)),
            dim=1,
        )
        return F.gelu(features + self.fuse(mixed))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.early = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.mid_context = MultiScaleBlock(64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.late_context = MultiScaleBlock(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._predict(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = []
        for row_offset, col_offset in (
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
        ):
            view = padded[
                :,
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            views.append(view)
            views.append(view.flip(-1))

        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
>>>>>>> REPLACE