MECHANISM: Coarse input-silhouette residual bypass

HYPOTHESIS: A zero-initialized classifier over 14×14 average-pooled input pixels will exceed 9,344 correct predictions by preserving coarse absolute shape and position information discarded by the trunk’s 30-dimensional bottleneck, with negligible runtime overhead.

INTENDED_EDIT: Add a normalized class-specific residual head over 2×2 average-pooled input images, increasing learned parameters from 247,418 to 249,780.

EVIDENCE: The normalized statistics bypass showed that direct class-specific paths around the bottleneck improve accuracy, while feature-map positional heads exceeded the time limit; pooling the single-channel input supplies positional evidence far more cheaply.

<<<<<<< SEARCH
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.silhouette_norm = nn.BatchNorm1d(14 * 14)
        self.silhouette_head = nn.Linear(14 * 14, 10)
        nn.init.zeros_(self.silhouette_head.weight)
        nn.init.zeros_(self.silhouette_head.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(feature_map) + residual_logits
=======
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        silhouette = F.avg_pool2d(images, kernel_size=2).flatten(1)
        silhouette_logits = self.silhouette_head(
            self.silhouette_norm(silhouette)
        )
        return self.classifier(feature_map) + residual_logits + silhouette_logits
>>>>>>> REPLACE