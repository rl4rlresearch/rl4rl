MECHANISM: Coarse spatial-pyramid classification head

HYPOTHESIS: Retaining 2×2 regional feature averages alongside global peaks will exceed the 9,327 correct predictions of Reference Design 3 by distinguishing classes whose local textures are similar but whose vertical and horizontal feature layouts differ.

INTENDED_EDIT: Replace the translation-invariant global mean/max head with a parameter-matched 2×2 spatial-average plus global-max head, while restoring Reference Design 3’s validated 22.5% flip-disagreement gating and calibration.

EVIDENCE: Reference Design 3 reached 9,327 correct using only global statistics, whereas explicit input-edge features fell to 9,286; this suggests the stronger alternative is preserving learned features’ coarse spatial arrangement rather than supplying more local edge channels. Paired offset training and multi-view inference can stabilize the new location-sensitive representation.

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
        regional_features = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((regional_features, peak_features), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
=======
        offset_weights = base_weights * (0.775 + 0.225 * flip_consistency)
>>>>>>> REPLACE

<<<<<<< SEARCH
        calibration = 1.24748 * (
=======
        calibration = 1.24754 * (
>>>>>>> REPLACE