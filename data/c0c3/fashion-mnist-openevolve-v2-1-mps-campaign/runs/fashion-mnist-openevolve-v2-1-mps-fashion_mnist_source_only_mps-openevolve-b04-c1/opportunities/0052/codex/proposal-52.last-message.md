MECHANISM: Zero-initialized coarse spatial logit bypass

HYPOTHESIS: Adding a direct 4×4 spatial readout to the successful fused-feature classifier will exceed 9,316 correct predictions by preserving class-specific location evidence alongside the nonlinear bottleneck.

INTENDED_EDIT: Add a zero-initialized adaptive-pooled linear branch from the fused feature map and sum its logits with the existing classifier, increasing parameters to 234,692 without changing the initial function.

EVIDENCE: The standalone multi-scale spatial readout reached 9,307 correct despite replacing the stronger 9,316-correct MLP, suggesting spatial evidence is complementary; the prior residual-bypass implementation was not verifiable, so this focused combination remains unresolved.

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
        self.spatial_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 10),
        )
        with torch.no_grad():
            self.spatial_classifier[-1].weight.zero_()
            self.spatial_classifier[-1].bias.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(fused)
=======
        return self.classifier(fused) + self.spatial_classifier(fused)
>>>>>>> REPLACE