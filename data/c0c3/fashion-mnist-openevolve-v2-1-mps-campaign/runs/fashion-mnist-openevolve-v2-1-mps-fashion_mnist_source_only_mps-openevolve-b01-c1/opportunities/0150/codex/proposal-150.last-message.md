MECHANISM: Coordinate-aware multi-scale part aggregation

HYPOTHESIS: Replacing the pixel-aligned flattened head with shared local part refinement plus coarse spatial-average and regional-maximum summaries will exceed 9,267 correct predictions by improving shift tolerance without discarding garment layout.

INTENDED_EDIT: Add a lightweight pointwise/depthwise residual part encoder and classify concatenated 4×4 average-pooled and 2×2 max-pooled features; the resulting model has approximately 249,178 learned parameters.

EVIDENCE: Position-free pooling regressed to 9,253 and 9,249, showing layout is essential, while widening the static flattened head regressed to 9,221 and attention timed out. This tests content-derived local parts and multi-scale aggregation while preserving explicit spatial regions at modest computational cost.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.part_projection = nn.Sequential(
            nn.Conv2d(56, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.part_context = nn.Sequential(
            nn.Conv2d(
                96,
                96,
                kernel_size=3,
                padding=1,
                groups=96,
                bias=False,
            ),
            nn.BatchNorm2d(96),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * (4 * 4 + 2 * 2), 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 10),
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
        parts = self.part_projection(features)
        parts = F.gelu(parts + self.part_context(parts))
        spatial_summary = torch.cat(
            (
                F.adaptive_avg_pool2d(parts, (4, 4)).flatten(1),
                F.adaptive_max_pool2d(parts, (2, 2)).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(spatial_summary)
>>>>>>> REPLACE