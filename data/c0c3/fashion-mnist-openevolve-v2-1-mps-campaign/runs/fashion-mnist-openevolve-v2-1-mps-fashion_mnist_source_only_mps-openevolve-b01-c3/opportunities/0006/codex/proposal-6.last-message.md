MECHANISM: Nine-position translation-and-flip test-time ensemble

HYPOTHESIS: Adding the four diagonal one-pixel translations to the best five-position ensemble will exceed 9,111 correct predictions because training exposes the residual model to joint horizontal and vertical translations.

INTENDED_EDIT: Replace the current CNN with the validated 240,654-parameter residual architecture and average evaluation logits across all nine one-pixel translation positions and their horizontal reflections.

EVIDENCE: The residual model improved from 9,094 to 9,111 correct when centered and cardinal translations were ensembled; diagonal views complete the local 3×3 translation neighborhood already represented by training augmentation.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(56, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
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
        for row_offset in range(3):
            for col_offset in range(3):
                view = padded[
                    :,
                    :,
                    row_offset : row_offset + height,
                    col_offset : col_offset + width,
                ]
                views.append(view)
                views.append(view.flip(-1))
        return self._predict(torch.cat(views, dim=0)).reshape(
            len(views), images.shape[0], 10
        ).mean(dim=0)
>>>>>>> REPLACE