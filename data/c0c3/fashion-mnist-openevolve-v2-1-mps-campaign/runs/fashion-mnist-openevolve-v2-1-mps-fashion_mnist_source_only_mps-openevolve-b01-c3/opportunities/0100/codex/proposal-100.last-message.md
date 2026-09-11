MECHANISM: Channel-adaptive mixed downsampling

HYPOTHESIS: Learning a per-channel blend of max and average pooling will exceed 9,287 correct predictions by reducing one-pixel phase sensitivity while preserving peak-sensitive channels.

INTENDED_EDIT: Replace both fixed max-pooling stages with channel-wise learned mixed pooling, reduce the classifier width by one unit to remain below 250,000 parameters, and retain the best-verified center-biased evaluation pooling and calibration.

EVIDENCE: Shift-view and center-weight refinements plateaued at 9,287 correct, showing that post-hoc aggregation is saturated. All available architectures assume winner-take-all max downsampling for every channel; learning which channels require smoother averaging directly targets the translation sensitivity that motivated the ten-view evaluation.

<<<<<<< SEARCH
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))


class ImageClassifier(nn.Module):
=======
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))


class AdaptiveDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_logits = nn.Parameter(
            torch.full((channels,), math.log(3.0))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_features = F.max_pool2d(features, 2)
        mean_features = F.avg_pool2d(features, 2)
        max_weight = torch.sigmoid(self.max_logits).view(1, -1, 1, 1)
        return (
            max_weight * max_features
            + (1.0 - max_weight) * mean_features
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pool1 = nn.MaxPool2d(2)
=======
        self.pool1 = AdaptiveDownsample(32)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pool2 = nn.MaxPool2d(2)
=======
        self.pool2 = AdaptiveDownsample(64)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
=======
            nn.Linear(192, 60),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(60, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        return 1.225 * view_logits.mean(dim=(0, 1))
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.546875
        return 1.22775 * pooled_logits
>>>>>>> REPLACE