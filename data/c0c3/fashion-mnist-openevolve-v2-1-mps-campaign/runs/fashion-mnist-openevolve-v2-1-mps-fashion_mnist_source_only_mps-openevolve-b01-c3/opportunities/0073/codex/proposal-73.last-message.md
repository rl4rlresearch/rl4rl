MECHANISM: Parameter-neutral quadrant-aware spatial-pyramid head

HYPOTHESIS: Retaining 2×2 average-pooled spatial layout alongside global peak evidence will exceed 9,287 correct predictions by distinguishing classes whose parts differ in arrangement.

INTENDED_EDIT: Expand the classifier input from global mean/max features to four quadrant averages plus global maxima, reducing its hidden width to 24 so the model remains below the parameter ceiling.

EVIDENCE: Tail-weighting variants plateaued at 9,287 correct and trimmed view pooling regressed to 9,276; the prior quadrant-aware proposal could not be verified, leaving this lightweight spatial mechanism unmeasured.

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
        quadrant_features = F.adaptive_avg_pool2d(features, 2).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((quadrant_features, peak_features), dim=1)
        )
>>>>>>> REPLACE