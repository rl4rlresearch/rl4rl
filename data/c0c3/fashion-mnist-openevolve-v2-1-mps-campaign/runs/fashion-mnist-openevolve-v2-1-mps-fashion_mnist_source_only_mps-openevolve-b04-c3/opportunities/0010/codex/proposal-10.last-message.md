MECHANISM: Dual-statistic coarse spatial pooling

HYPOTHESIS: With the qualified batch-64 encoder and training procedure, replacing the position-specific 7×7 flattening bottleneck with 2×2 average/max pooling and a 235-unit fusion layer will exceed 9,229 correct predictions by learning broader shape-and-presence interactions without materially increasing runtime.

INTENDED_EDIT: Challenge the assumption that preserving every final spatial coordinate through a narrow 38-unit layer is the best parameter use; instead, aggregate each feature channel into coarse average and maximum maps, then classify their concatenation with a much wider head. The resulting model has approximately 249,329 learned parameters.

EVIDENCE: Reference Design 2 established 9,229 correct predictions for this encoder at batch size 64, while adding spatial residual computation and compute-light capacity on top both timed out. This replacement keeps encoder computation unchanged and substitutes—not adds—a similarly sized classifier with a different spatial representation.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
=======
        self.classifier = nn.Sequential(
            nn.Linear(96 * 2 * 2 * 2, 235),
            nn.LayerNorm(235),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(235, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        average_features = F.adaptive_avg_pool2d(features, output_size=2)
        maximum_features = F.adaptive_max_pool2d(features, output_size=2)
        pooled = torch.cat((average_features, maximum_features), dim=1)
        return self.classifier(pooled.flatten(1))
>>>>>>> REPLACE