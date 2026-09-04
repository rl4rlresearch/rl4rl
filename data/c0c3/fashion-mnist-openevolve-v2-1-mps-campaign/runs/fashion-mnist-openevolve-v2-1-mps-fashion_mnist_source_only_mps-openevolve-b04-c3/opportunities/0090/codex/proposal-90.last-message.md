MECHANISM: Multi-resolution channel-factorized spatial head

HYPOTHESIS: A learned 24-channel local projection plus full-channel global summary feeding a 141-unit nonlinear head will exceed 9,243 correct predictions within the parameter ceiling.

INTENDED_EDIT: Replace the 4,704-to-38 bottleneck with a shared per-location channel compressor, retain all 7×7 locations, concatenate a 96-channel global summary, and classify through a substantially wider latent representation.

EVIDENCE: The current design spends 178,752 weights collapsing all spatial-channel features into only 38 units, while the 9,166-correct attention result indicates that discarding spatial detail is harmful. This 249,331-parameter alternative preserves every spatial location and avoids the failed direct-logit shortcut by learning nonlinear interactions through a 141-unit head.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
=======
        self.local_projection = nn.Sequential(
            nn.Conv2d(96, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(24 * 7 * 7 + 96, 141),
            nn.LayerNorm(141),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(141, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        local_features = self.local_projection(features).flatten(1)
        global_features = features.mean(dim=(-2, -1))
        return self.classifier(
            torch.cat((local_features, global_features), dim=1)
        )
>>>>>>> REPLACE