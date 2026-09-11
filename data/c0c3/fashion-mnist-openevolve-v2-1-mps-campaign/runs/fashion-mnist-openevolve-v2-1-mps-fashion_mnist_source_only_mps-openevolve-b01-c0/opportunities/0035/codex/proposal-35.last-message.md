MECHANISM: Zero-initialized nonlinear residual classification adapter

HYPOTHESIS: Adding a 30-unit residual MLP branch to the validated 58-unit head will exceed 9,290 correct predictions by increasing class-separation capacity without reducing convolutional width or disrupting the initial direct classifier.

INTENDED_EDIT: Preserve the existing direct output layer and add a zero-initialized BatchNorm-SiLU adapter branch, bringing the model to 249,936 learned parameters without changing training or evaluation cost materially.

EVIDENCE: Expanding the dense bottleneck from 48 to 58 produced the 9,290-correct best design, whereas expanding it further by reducing convolutional width fell to 9,264; this motivates spending the remaining parameter budget on the head while preserving all validated convolutional capacity.

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
        )
        self.output = nn.Linear(58, 10)
        self.output_adapter = nn.Sequential(
            nn.Linear(58, 30, bias=False),
            nn.BatchNorm1d(30),
            nn.SiLU(),
            nn.Linear(30, 10, bias=False),
        )
        nn.init.zeros_(self.output_adapter[-1].weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.classifier(self.features(images))
        return self.output(features) + self.output_adapter(features)
>>>>>>> REPLACE