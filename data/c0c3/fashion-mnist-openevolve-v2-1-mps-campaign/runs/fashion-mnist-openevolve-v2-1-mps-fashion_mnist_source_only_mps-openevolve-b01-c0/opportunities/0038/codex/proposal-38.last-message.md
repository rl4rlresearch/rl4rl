MECHANISM: Dual-statistic global residual classifier

HYPOTHESIS: Adding a zero-initialized global mean/max classification branch will exceed 9,290 correct predictions by providing translation-invariant class evidence while preserving the validated convolutional widths and primary head.

INTENDED_EDIT: Add a compute-light 1,290-parameter auxiliary classifier over global mean and maximum feature statistics, raising the model to 249,126 parameters.

EVIDENCE: Expanding the dense head from 48 to 58 units produced the 9,290-correct best result, while sacrificing convolutional width for further expansion regressed to 9,264; the residual adapter timed out, motivating a cheaper complementary head that retains all validated channels.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.BatchNorm1d(58),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(58, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.BatchNorm1d(58),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(58, 10),
        )
        self.global_classifier = nn.Linear(64 * 2, 10)
        nn.init.zeros_(self.global_classifier.weight)
        nn.init.zeros_(self.global_classifier.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        global_statistics = torch.cat(
            (
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
            ),
            dim=1,
        )
        return (
            self.classifier(features)
            + self.global_classifier(global_statistics)
        )
>>>>>>> REPLACE