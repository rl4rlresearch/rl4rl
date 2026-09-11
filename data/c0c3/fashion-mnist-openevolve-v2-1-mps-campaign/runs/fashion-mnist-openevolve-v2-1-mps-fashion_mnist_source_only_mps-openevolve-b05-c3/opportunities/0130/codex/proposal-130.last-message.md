MECHANISM: Parallel class-conditioned spatial evidence pooling

HYPOTHESIS: A zero-initialized class-specific attention head over the final 3×3 feature map will exceed 9,206 correct predictions by learning where each class’s evidence occurs, while verified 1.10× outer calibration lowers tied-count cross-entropy.

INTENDED_EDIT: Preserve the qualified residual training system, add a low-cost attention-pooled class-evidence path alongside the shared flattened classifier, and use the verified 1.10× post-ensemble sharpening.

EVIDENCE: The 9,206-correct designs assume every prediction passes through one shared 48-unit flattened bottleneck. The late global-moment branch reached only 9,201, indicating uniform global summaries are insufficient; class-conditioned pooling retains spatial selection. Unlike the timed-out hierarchical attention design, this operates on only nine final tokens and adds just 1,940 parameters and one 1×1 convolution.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
        self.spatial_head = nn.Conv2d(96, 20, kernel_size=1)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        dense_logits = self.classifier(features)
        evidence, attention_logits = self.spatial_head(features).flatten(2).chunk(
            2, dim=1
        )
        attention = F.softmax(attention_logits, dim=2)
        spatial_logits = (evidence * attention).sum(dim=2)
        return dense_logits + spatial_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE