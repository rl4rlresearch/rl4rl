MECHANISM: Hybrid spatial-shape and second-order channel-covariance classifier

HYPOTHESIS: Retaining nearly the full spatial bottleneck while adding a learned covariance branch will increase validation_correct above 9,243 by capturing texture and channel co-occurrence distinctions that the current flattened first-order head cannot represent.

INTENDED_EDIT: Reduce the existing hidden width from 38 to 37 and use the recovered parameter budget for a 24-channel centered covariance classifier, producing a 249,597-parameter model with minimal additional computation.

EVIDENCE: Repeated temperature and ensemble refinements plateaued at exactly 9,243 correct, indicating that calibration is no longer the limiting assumption. Unlike the unverifiable fixed 4×2 pooling experiment, this preserves the established spatial prediction path almost intact while testing a genuinely different, orderless second-order representation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 37),
            nn.LayerNorm(37),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(37, 10),
        )
        self.bilinear_projection = nn.Sequential(
            nn.Conv2d(96, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.bilinear_classifier = nn.Sequential(
            nn.LayerNorm(24 * 24),
            nn.Linear(24 * 24, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
=======
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        spatial_logits = self.classifier(features)

        projected = self.bilinear_projection(features).flatten(2)
        centered = projected - projected.mean(dim=2, keepdim=True)
        covariance = torch.bmm(centered, centered.transpose(1, 2))
        covariance = covariance / (centered.shape[2] - 1)
        bilinear_logits = self.bilinear_classifier(covariance.flatten(1))

        return spatial_logits + bilinear_logits
>>>>>>> REPLACE