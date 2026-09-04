MECHANISM: Coarse-to-global spatial pyramid decoder

HYPOTHESIS: Replacing fixed 7×7 flattening with pooled 4×4, 2×2, and global representations will exceed 9,247 correct predictions by preserving coarse garment layout while reducing sensitivity to crop-induced feature displacement.

INTENDED_EDIT: Decode the existing 48-channel feature map through multiscale average pooling and a wider 200-unit classifier, using 245,442 learned parameters.

EVIDENCE: A larger flattened head regressed to 9,210 and phase-preserving space-to-depth regressed to 9,191, indicating that more fixed-coordinate detail is unhelpful; this efficiently tests hierarchical spatial aggregation without the covariance decoder’s prohibitive runtime.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Linear(48 * (1 + 4 + 16), 200),
            nn.GELU(),
            nn.LayerNorm(200),
            nn.Linear(200, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        pyramid = torch.cat(
            (
                F.adaptive_avg_pool2d(features, 4).flatten(1),
                F.adaptive_avg_pool2d(features, 2).flatten(1),
                F.adaptive_avg_pool2d(features, 1).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(pyramid)
>>>>>>> REPLACE