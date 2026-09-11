MECHANISM: Dual-path late spatial consolidation

HYPOTHESIS: Combining learned stride-2 aggregation with a parameter-free averaged identity path will exceed 9,255 correct predictions by preserving coarse channel evidence that the best verified single-path aggregator may discard.

INTENDED_EDIT: Replace the flat 7×7 classifier with a 48-channel learned late aggregator concatenated with 80-channel 3×3 average-pooled features, use a 76-unit nonlinear head, and adopt the best architecture’s verified 1.253 calibration; the model has 245,054 learned parameters.

EVIDENCE: Learned late spatial consolidation improved correctness from 9,240 to 9,255, whereas replacing early pooling reduced it to 9,208. This patch retains the successful learned late downsampling and adds pooling only as a complementary identity path.

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
                80, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((48 + 80) * 4 * 4, 76),
            nn.LayerNorm(76),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(76, 10),
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
        aggregated = self.aggregate(features)
        pooled = F.avg_pool2d(
            features, kernel_size=3, stride=2, padding=1
        )
        return self.classifier(torch.cat((aggregated, pooled), dim=1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.285 * logits
=======
        return 1.253 * logits
>>>>>>> REPLACE