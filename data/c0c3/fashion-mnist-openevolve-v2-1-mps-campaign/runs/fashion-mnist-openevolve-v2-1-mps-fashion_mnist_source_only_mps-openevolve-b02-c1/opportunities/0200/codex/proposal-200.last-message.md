MECHANISM: Content-addressed multi-part spatial pooling

HYPOTHESIS: Replacing the rigid 3×3 flattened representation with global statistics plus three learned spatial part tokens will exceed 9,324 correct predictions by recognizing discriminative features independently of their exact location.

INTENDED_EDIT: Remove the final spatial downsampling and fixed-position flattening head; pool the 7×7 feature map into global mean, global maximum, and three learned attention-weighted part descriptors before classification. This challenges the old assumption that semantic features must occupy fixed grid cells.

EVIDENCE: Adaptive max pooling fell to 9,314, showing that merely repairing pooling geometry is insufficient, while optimization refinements around the fixed flattened representation plateaued at 9,324. Learned part pooling uses every spatial feature while preserving several distinct content-selected summaries, providing a more expressive alternative to both rigid flattening and single-statistic pooling.

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
        ensemble = logits * 2.25
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.25
            ensemble = ensemble + flipped_logits
        return ensemble / (12.5 * 0.9350)
=======
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
        )
        self.spatial_attention = nn.Conv2d(96, 3, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(96 * 5, 260),
            nn.BatchNorm1d(260),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(260, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        spatial_features = features.flatten(2)
        attention = self.spatial_attention(features).flatten(2)
        attention = attention.softmax(dim=-1)
        parts = torch.bmm(attention, spatial_features.transpose(1, 2))
        descriptor = torch.cat(
            (
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
                parts.flatten(1),
            ),
            dim=1,
        )
        return self.classifier(descriptor)

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
        ensemble = logits * 2.25
        for view in views[1:]:
            ensemble = ensemble + self._predict(view)
        for view_index, view in enumerate(views):
            flipped_logits = self._predict(view.flip(-1))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.25
            ensemble = ensemble + flipped_logits
        return ensemble / (12.5 * 0.9350)
>>>>>>> REPLACE