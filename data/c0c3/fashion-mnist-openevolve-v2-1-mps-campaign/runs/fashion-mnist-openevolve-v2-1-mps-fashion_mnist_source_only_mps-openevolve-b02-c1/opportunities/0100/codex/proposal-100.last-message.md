MECHANISM: Centered bilinear covariance pooling with coarse spatial classification

HYPOTHESIS: Replacing the lossy positional dense head with a covariance-based second-order branch plus a 2×2 spatial branch will exceed 9,286 validation-correct predictions while remaining within the time and parameter limits.

INTENDED_EDIT: Preserve the terminal 7×7 feature map, classify learned channel co-occurrences through normalized covariance pooling, and combine them with a compact coarse-layout MLP; training, augmentation, averaging, and TTA remain unchanged.

EVIDENCE: Overlapping terminal max-pooling regressed to 9,234 and TTA tuning plateaued at 9,286, challenging the assumption that stronger decisions come from the same lossy 3×3 positional representation. This 238,132-parameter alternative tests quadratic feature relationships without adding convolutional branches or their observed timeout risk.

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.bilinear_classifier = nn.Linear(96 * 96, 10)
        self.spatial_classifier = nn.Sequential(
            nn.Linear(96 * 2 * 2, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        descriptors = feature_map.flatten(2)
        centered = descriptors - descriptors.mean(dim=2, keepdim=True)
        covariance = torch.bmm(centered, centered.transpose(1, 2))
        covariance = covariance / max(centered.shape[2] - 1, 1)
        bilinear = covariance.flatten(1)
        bilinear = torch.sign(bilinear) * torch.sqrt(bilinear.abs() + 1e-6)
        bilinear = F.normalize(bilinear, dim=1)

        coarse_spatial = F.adaptive_avg_pool2d(feature_map, 2).flatten(1)
        return (
            self.bilinear_classifier(bilinear)
            + self.spatial_classifier(coarse_spatial)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
=======
        for view in views[1:]:
            view_logits = self._classify(view)
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self._classify(view.flip(-1))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
>>>>>>> REPLACE