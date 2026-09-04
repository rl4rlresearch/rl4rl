MECHANISM: Channel-specific separable spatial attention pooling

HYPOTHESIS: Replacing global mean pooling with per-channel learned 7×7 spatial attention will exceed 9,285 correct predictions by preserving class-relevant feature layout, while linear final-10% averaging retains the strongest verified optimization baseline.

INTENDED_EDIT: Add zero-initialized separable row/column attention for each late feature channel, reduce the classifier width to remain below 250,000 parameters, and restore the verified final-10% linearly recency-weighted parameter average.

EVIDENCE: Final-10% linear recency weighting achieved the best result at 9,285 correct; spatial refinement and evaluation-matched translations showed that spatial structure matters, yet the current global mean statistic discards feature location entirely.

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
        self.row_attention = nn.Parameter(torch.zeros(96, 7))
        self.col_attention = nn.Parameter(torch.zeros(96, 7))
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 54),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(54, 10),
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
        spatial_logits = (
            self.row_attention.unsqueeze(2)
            + self.col_attention.unsqueeze(1)
        )
        spatial_weights = F.softmax(
            spatial_logits.flatten(1), dim=1
        ).reshape(1, 96, 7, 7)
        attended_features = (features * spatial_weights).sum(dim=(2, 3))
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((attended_features, peak_features), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.95 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
        average_weight = 1.0 / optimizer.tail_average_count
=======
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
>>>>>>> REPLACE