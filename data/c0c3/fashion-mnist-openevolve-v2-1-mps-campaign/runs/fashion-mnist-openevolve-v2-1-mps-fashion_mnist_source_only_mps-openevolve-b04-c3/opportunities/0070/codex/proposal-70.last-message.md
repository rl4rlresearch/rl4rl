MECHANISM: Spatially preserved factorized convolutional head

HYPOTHESIS: Replacing the 38-unit monolithic flattened bottleneck with local channel projection, residual refinement, and a 110-unit spatial reasoning layer will exceed 9,243 correct predictions within 244,280 parameters.

INTENDED_EDIT: Preserve the complete 7×7 feature grid, project its channels from 96 to 32, refine them locally, and use a nearly three-times-wider global latent representation before classification.

EVIDENCE: The current 245,040-parameter model reaches 9,243 correct while spending 179,256 weights on a single flatten-to-38 layer. The 9,166-correct attention design showed that aggressively pooling spatial features is harmful; this alternative reallocates those parameters without pooling away any locations.

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
        self.feature_projection = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualDepthwiseBlock(32),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 110),
            nn.LayerNorm(110),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(110, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_projection(self.stem(images))
        return self.classifier(features)
>>>>>>> REPLACE