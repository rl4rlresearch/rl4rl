MECHANISM: Residual bilinear covariance pooling

HYPOTHESIS: Adding a normalized channel-covariance classifier will exceed 9,257 correct predictions by capturing position-robust feature co-occurrences that the coordinate-specific flattened head and failed first-order global-average branch cannot represent.

INTENDED_EDIT: Compute the upper-triangular covariance of the 56 feature channels, apply signed-square-root and L2 normalization, and add a zero-initialized ten-class bilinear readout to the existing logits; total learned parameters become 249,404.

EVIDENCE: The global-average residual reached only 9,253 correct and additional local depth reached 9,228, suggesting neither first-order pooled evidence nor more local processing supplies the missing discrimination. Full covariance pooling instead introduces second-order channel interactions while preserving the best 233,434-parameter backbone and spatial classifier.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        covariance_indices = torch.triu_indices(56, 56)
        self.register_buffer(
            "covariance_rows", covariance_indices[0], persistent=False
        )
        self.register_buffer(
            "covariance_cols", covariance_indices[1], persistent=False
        )
        self.covariance_classifier = nn.Linear(56 * 57 // 2, 10)
        nn.init.zeros_(self.covariance_classifier.weight)
        nn.init.zeros_(self.covariance_classifier.bias)
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
        spatial_logits = self.classifier(features)

        centered = features.flatten(2)
        centered = centered - centered.mean(dim=-1, keepdim=True)
        covariance = torch.bmm(centered, centered.transpose(1, 2))
        covariance = covariance / centered.shape[-1]
        moments = covariance[
            :, self.covariance_rows, self.covariance_cols
        ]
        moments = torch.sign(moments) * torch.sqrt(
            moments.abs() + 1.0e-5
        )
        moments = F.normalize(moments, p=2.0, dim=1)
        return spatial_logits + self.covariance_classifier(moments)
>>>>>>> REPLACE