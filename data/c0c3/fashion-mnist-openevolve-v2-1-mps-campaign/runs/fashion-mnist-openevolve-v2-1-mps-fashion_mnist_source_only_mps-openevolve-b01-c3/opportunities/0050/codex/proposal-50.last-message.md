MECHANISM: Learned axial-profile classification head

HYPOTHESIS: Replacing purely translation-invariant global pooling with learned channel projections summarized as row and column activation profiles will exceed 9,285 correct predictions by exposing garment-part arrangement while retaining robust global mean and peak features.

INTENDED_EDIT: Challenge the assumption that global mean/max statistics contain sufficient shape information: project late features into 16 layout channels, preserve their seven-row and seven-column profiles, and classify the combined global-and-layout representation through a parameter-budgeted head.

EVIDENCE: The strongest global-pooling design reached 9,285 correct, while changing its averaging density moved accuracy by only 2–4 images and repeated averaging variants timed out. Local spatial refinement improved correctness from 9,202 to 9,209, indicating spatial organization is useful; axial profiles test that signal directly without costly all-pairs attention and remain under 250,000 parameters.

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
        self.layout_projection = nn.Conv2d(
            96, 16, kernel_size=1, bias=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(416),
            nn.Linear(416, 24),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(24, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)

        layout_features = F.gelu(self.layout_projection(features))
        row_profiles = layout_features.mean(dim=3).flatten(1)
        column_profiles = layout_features.mean(dim=2).flatten(1)
        return self.classifier(
            torch.cat(
                (
                    mean_features,
                    peak_features,
                    row_profiles,
                    column_profiles,
                ),
                dim=1,
            )
        )
>>>>>>> REPLACE