MECHANISM: Zero-initialized coarse spatial logit bypass

HYPOTHESIS: Adding a direct 4×4 spatial readout alongside the existing fused-feature classifier will exceed 9,316 correct predictions by retaining complementary class-specific location evidence.

INTENDED_EDIT: Add a zero-initialized adaptive-pooled linear branch from the fused feature map and sum its logits with the existing classifier output, raising parameters from 224,442 to 234,692.

EVIDENCE: The standalone multi-scale spatial readout achieved 9,307 correct despite replacing the stronger 9,316-correct classifier, indicating spatial evidence is promising and potentially complementary; the prior bypass attempts were not verifiable, so the combination remains unresolved.

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
        self.spatial_classifier = nn.Linear(64 * 4 * 4, 10)
        nn.init.zeros_(self.spatial_classifier.weight)
        nn.init.zeros_(self.spatial_classifier.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        spatial_features = F.adaptive_avg_pool2d(
            fused, output_size=(4, 4)
        ).flatten(1)
        return (
            self.classifier(fused)
            + self.spatial_classifier(spatial_features)
        )
>>>>>>> REPLACE