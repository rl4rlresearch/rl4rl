MECHANISM: Learned late spatial consolidation

HYPOTHESIS: Replacing absolute 7×7 flattening with a shared learned 3×3 stride-2 aggregation and a wider nonlinear head will exceed 9,240 correct predictions by encoding local garment-part relationships while retaining coarse 4×4 layout.

INTENDED_EDIT: Challenge the assumption that class prediction should independently weight every 7×7 location. Add a learned late spatial aggregator, widen the classifier bottleneck from 48 to 140, and retain the best verified smoothing and 1.253 calibration; the resulting model has 245,150 learned parameters.

EVIDENCE: A second same-grid refinement block fell to 9,232, while fixed overlapping early pooling fell to 9,208. This motivates reallocating dense-head parameters to learned late downsampling that preserves the successful early feature extractor instead of adding same-resolution processing or destroying high-resolution evidence.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
=======
        self.aggregate = nn.Sequential(
            nn.Conv2d(
                80, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = self.aggregate(features)
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.255 * logits
=======
        return 1.253 * logits
>>>>>>> REPLACE