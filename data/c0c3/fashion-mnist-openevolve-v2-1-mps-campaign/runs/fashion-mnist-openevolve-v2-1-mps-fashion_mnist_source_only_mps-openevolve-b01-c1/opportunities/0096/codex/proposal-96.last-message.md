MECHANISM: Coarse-to-global spatial pyramid decoder

HYPOTHESIS: Replacing the fixed 7×7 representation with pooled 4×4, 2×2, and global features will exceed 9,247 correct predictions by retaining garment layout while reducing crop-position sensitivity.

INTENDED_EDIT: Concatenate three average-pooled feature scales and classify them with a 200-unit head, totaling 245,442 learned parameters.

EVIDENCE: Enlarging the fixed-coordinate head regressed to 9,210 correct, whereas the prior multiscale pooling proposal could not be verified and therefore remains untested.

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
            nn.Linear(48 * (4 * 4 + 2 * 2 + 1), 200),
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
        spatial_pyramid = torch.cat(
            (
                F.adaptive_avg_pool2d(features, 4).flatten(1),
                F.adaptive_avg_pool2d(features, 2).flatten(1),
                F.adaptive_avg_pool2d(features, 1).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(spatial_pyramid)
>>>>>>> REPLACE