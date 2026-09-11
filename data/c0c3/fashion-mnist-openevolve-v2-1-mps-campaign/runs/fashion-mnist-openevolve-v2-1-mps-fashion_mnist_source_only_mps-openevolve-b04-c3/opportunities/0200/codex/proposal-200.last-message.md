MECHANISM: Dual-path spatial and activation-statistics evidence head

HYPOTHESIS: Replacing the single 4,704→38 bottleneck with a near-capacity spatial pathway plus a complementary learned mean/max/RMS pathway will exceed 9,251 correct predictions while remaining under the parameter and runtime limits.

INTENDED_EDIT: Preserve the proven convolutional trunk and training procedure, but compute predictions from jointly learned position-sensitive features and translation-insensitive activation statistics; the resulting model has approximately 244,735 learned parameters.

EVIDENCE: Fine-grained EMA blending repeatedly plateaued at 9,251 correct, showing that calibration cannot repair the current model’s errors, while the six-block alternative timed out. The load-bearing old assumption is that all class evidence should pass through one flattened bottleneck; this patch tests complementary global evidence extraction with slightly less computation than the current head.

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
        self.spatial_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.statistics_head = nn.Sequential(
            nn.Linear(96 * 3, 46),
            nn.LayerNorm(46),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.classifier = nn.Linear(35 + 46, 10)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
=======
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        spatial = self.spatial_head(features)

        mean = features.mean(dim=(2, 3))
        maximum = features.amax(dim=(2, 3))
        rms = features.square().mean(dim=(2, 3)).add(1e-6).sqrt()
        statistics = self.statistics_head(torch.cat((mean, maximum, rms), dim=1))

        return self.classifier(torch.cat((spatial, statistics), dim=1))
>>>>>>> REPLACE