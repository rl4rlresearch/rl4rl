MECHANISM: Dual-statistic spatial-pyramid representation

HYPOTHESIS: Replacing absolute 7×7 flattening with multi-scale average-and-max pooled features will exceed 9,239 correct predictions by preserving coarse garment layout and localized feature presence while reducing sensitivity to exact spatial coordinates.

INTENDED_EDIT: Reallocate the dense head into a 1×1/2×2/4×4 dual-statistic spatial pyramid and a 100-unit classifier, yielding 244,142 learned parameters with similar dense-head compute.

EVIDENCE: The coordinate-specific head remains the shared load-bearing assumption; widening it regressed to 9,210, and adding spatial convolution regressed to 9,188. The timed-out semantic-token design confounded adaptive pooling with a deeper backbone, whereas this patch cleanly tests position-tolerant image representation without additional learned feature-map operations.

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
            nn.Linear(48 * 2 * (1 + 4 + 16), 100),
            nn.GELU(),
            nn.LayerNorm(100),
            nn.Linear(100, 10),
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
            [
                pooled.flatten(1)
                for level in (1, 2, 4)
                for pooled in (
                    F.adaptive_avg_pool2d(features, level),
                    F.adaptive_max_pool2d(features, level),
                )
            ],
            dim=1,
        )
        return self.classifier(pyramid)
>>>>>>> REPLACE