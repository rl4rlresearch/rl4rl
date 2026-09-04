MECHANISM: Parameter-neutral vertical spatial pooling head

HYPOTHESIS: Preserving separate upper- and lower-image feature averages while retaining global peak features will exceed 9,287 correct predictions by capturing class-relevant vertical layout discarded by global mean pooling.

INTENDED_EDIT: Replace global mean features with 2×1 vertical pooled features and resize the classifier from 192→61 to 288→40, yielding approximately 249,730 learned parameters while retaining the verified evaluation ensemble.

EVIDENCE: Evaluation-pooling refinements plateaued at 9,287 correct, while the coarse-spatial-head attempt could not be verified; this tests the unresolved spatial-layout hypothesis without its added 1×1 projection and preserves more hidden width than a full 2×2 representation.

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
            nn.LayerNorm(288),
            nn.Linear(288, 40),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(40, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        vertical_features = F.adaptive_avg_pool2d(features, (2, 1)).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(
            torch.cat((vertical_features, peak_features), dim=1)
        )
>>>>>>> REPLACE