MECHANISM: Learned coarse-spatial projection head

HYPOTHESIS: Replacing global mean pooling with a learned 32-channel 2×2 spatial representation while retaining global peak features will exceed 9,287 correct predictions by preserving class-relevant feature arrangement that the current global pooling discards.

INTENDED_EDIT: Add a lightweight nonlinear 1×1 feature projection, pool it into four spatial regions, and reallocate the classifier width to remain under the 250,000-parameter ceiling.

EVIDENCE: View-weight refinements plateaued at 9,287 correct and primarily changed cross-entropy, while stronger weighting lost a prediction; this suggests the load-bearing limitation is the shared assumption that all spatial layout should be erased before classification.

<<<<<<< SEARCH
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(224),
            nn.Linear(224, 39),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(39, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        spatial_features = F.adaptive_avg_pool2d(
            self.spatial_projection(features),
            (2, 2),
        ).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((spatial_features, peak_features), dim=1)
        )
>>>>>>> REPLACE