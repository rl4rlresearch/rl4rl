MECHANISM: Coarse spatial logit bypass

HYPOTHESIS: Adding a direct 4×4 spatial readout alongside the successful nonlinear classifier will exceed 9,316 correct predictions by preserving class-specific coarse location evidence without replacing the proven fused-feature pathway.

INTENDED_EDIT: Add a zero-initialized, adaptive-pooled linear logit branch from the fused feature map and sum it with the existing classifier output, increasing parameters from 224,442 to 234,692 with negligible computation.

EVIDENCE: The multi-scale convolutional evidence readout reached 9,307 correct—close to the 9,316 baseline despite discarding its successful MLP—suggesting direct spatial evidence is useful but insufficient alone; a residual bypass combines that evidence with the stronger existing classifier.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
        self.spatial_skip = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 10),
        )
        with torch.no_grad():
            self.spatial_skip[-1].weight.zero_()
            self.spatial_skip[-1].bias.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(fused)
=======
        return self.classifier(fused) + self.spatial_skip(fused)
>>>>>>> REPLACE