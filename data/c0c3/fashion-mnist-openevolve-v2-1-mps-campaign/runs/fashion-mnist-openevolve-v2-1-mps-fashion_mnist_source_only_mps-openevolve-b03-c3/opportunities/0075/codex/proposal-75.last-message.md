MECHANISM: Interpolated local-context late consolidation

HYPOTHESIS: A 44-local/20-dilated channel split will exceed 9,268 correct predictions by retaining more local detail than the failed 32/32 split while adding more context than the successful 48/16 split.

INTENDED_EDIT: Replace the flat classifier with parallel 44-channel local and 20-channel dilation-2 aggregators feeding the verified 140-unit head; preserve all training, augmentation, TTA, and calibration settings.

EVIDENCE: Correctness rose from 9,255 with local-only aggregation to 9,265 at 48/16 and 9,268 at 40/24, then fell sharply to 9,243 at 32/32. Testing 44/20 targets the unmeasured interval around the apparent optimum while keeping 245,150 parameters.

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
        self.aggregate_local = nn.Sequential(
            nn.Conv2d(
                80, 44, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(44),
            nn.GELU(),
        )
        self.aggregate_context = nn.Sequential(
            nn.Conv2d(
                80,
                20,
                kernel_size=3,
                stride=2,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(20),
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
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE