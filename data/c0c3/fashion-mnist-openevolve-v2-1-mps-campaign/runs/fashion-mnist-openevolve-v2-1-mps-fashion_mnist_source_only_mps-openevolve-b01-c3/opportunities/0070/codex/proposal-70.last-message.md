MECHANISM: Quadrant-aware spatial-pyramid classification head

HYPOTHESIS: Preserving four coarse spatial regions alongside global peak evidence will exceed 9,287 correct predictions by distinguishing classes through part arrangement that the current globally pooled representation discards.

INTENDED_EDIT: Replace global mean/max classification with a parameter-neutral 2×2 average-pyramid plus global-max head, and restore the strongest verified linear-recency tail averaging baseline.

EVIDENCE: Ten-view pooling with linear endpoint quadrature reached 9,287 correct, while further averaging and view-pooling changes did not improve it; spatial mechanisms remain plausible but prior attention-based versions timed out, motivating this lightweight learned use of coarse layout.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.LayerNorm(480),
            nn.Linear(480, 24),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(24, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        spatial_features = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((spatial_features, peak_features), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        recency_power = 1.5
=======
        recency_power = 1.0
>>>>>>> REPLACE