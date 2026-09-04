MECHANISM: Identity-initialized channel recalibration

HYPOTHESIS: Adding lightweight squeeze-excitation to the verified 39/64 curriculum will exceed 9,167 correct predictions by adaptively reweighting convolutional channels without sacrificing the position-sensitive 48-unit head.

INTENDED_EDIT: Restore the best verified augmentation boundary and spend the remaining parameter budget on an identity-initialized 96→16→96 channel gate, producing 249,002 learned parameters.

EVIDENCE: The 39/64 curriculum with the 48-unit head achieved the best result of 9,167 correct, while widening that head to 52 units reduced performance to 9,164; this motivates allocating spare capacity to feature recalibration instead of further head widening.

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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 96),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.channel_gate[-2].weight)
        nn.init.zeros_(self.channel_gate[-2].bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        channel_scale = self.channel_gate(features).view(-1, 96, 1, 1)
        return self.classifier(features * (2.0 * channel_scale))
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 3 < total_steps * 2:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE