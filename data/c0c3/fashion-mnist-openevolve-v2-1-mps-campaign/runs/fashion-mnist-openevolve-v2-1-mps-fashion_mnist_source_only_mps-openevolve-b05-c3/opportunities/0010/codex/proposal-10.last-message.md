MECHANISM: Class-specific spatial attention pooling

HYPOTHESIS: Replacing the spatially indexed flatten-MLP head with class-specific attention over local evidence, while reallocating its parameters to 112-channel convolutional features, will exceed 9,128 correct predictions by reducing sensitivity to the final 3×3 grid alignment without discarding localized garment cues.

INTENDED_EDIT: Replace the old assumption that a flattened 3×3 feature grid is the best classifier input with two fully convolutional heads that learn class-specific evidence and attention maps; widen the final feature stages while retaining the validated batch-64 training and ten-view inference.

EVIDENCE: Cardinal translation ensembling improved flip-only evaluation from 9,110 to 9,125 correct, showing that spatial alignment remains load-bearing, while diagonal views reduced accuracy; learned attention pooling directly targets that sensitivity instead of adding more displaced inference views.

<<<<<<< SEARCH
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
=======
        channels = (
            (1, 32),
            (32, 32),
            (32, 64),
            (64, 64),
            (64, 112),
            (112, 112),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.feature_dropout = nn.Dropout2d(0.10)
        self.class_evidence = nn.Conv2d(112, 10, kernel_size=1)
        self.class_attention = nn.Conv2d(112, 10, kernel_size=1)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_dropout(self.features(images))
        evidence = self.class_evidence(features).flatten(2)
        attention = F.softmax(
            self.class_attention(features).flatten(2),
            dim=2,
        )
        return (evidence * attention).sum(dim=2)
>>>>>>> REPLACE