MECHANISM: Identity-initialized post-pool residual refinement

HYPOTHESIS: A low-cost bottleneck residual block plus a 54-unit classifier will increase validation_correct above 9,252 while remaining below the 250,000-parameter ceiling.

INTENDED_EDIT: Add an identity-initialized 64→32→64 residual block at 7×7 resolution and widen the classifier bottleneck from 48 to 54 units, yielding 248,808 learned parameters.

EVIDENCE: The verified design reached 9,252 correct with 33,654 parameters unused; inference-only changes cannot improve its argmax, while previous training-side trials timed out without providing contrary accuracy evidence.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        nn.init.zeros_(self.refine[-1].weight)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 54),
            nn.BatchNorm1d(54),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(54, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.relu(features + self.refine(features), inplace=True)
        return self.classifier(features)
>>>>>>> REPLACE