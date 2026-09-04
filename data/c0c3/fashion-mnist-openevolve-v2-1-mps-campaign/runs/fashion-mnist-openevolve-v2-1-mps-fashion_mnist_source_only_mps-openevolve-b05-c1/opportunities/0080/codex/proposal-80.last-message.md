MECHANISM: Low-rank bilinear channel-covariance pooling

HYPOTHESIS: A learned 12-dimensional covariance descriptor will exceed 9,345 correct predictions by exposing cross-channel co-occurrence evidence that the flattened bottleneck and independent mean/std/max statistics cannot represent.

INTENDED_EDIT: Add a zero-initialized residual classifier over the upper triangle of a learned low-rank channel covariance matrix, preserving the current predictor at initialization and remaining below the parameter ceiling.

EVIDENCE: The first-order statistics bypass improved the classifier, but diagonal quadratic enrichment fell to 9,338, indicating that additional per-channel curvature is insufficient; this patch instead tests genuine cross-channel interactions after cheap 7×7 aggregation.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.covariance_projection = nn.Conv2d(
            64, 12, kernel_size=1, bias=False
        )
        self.register_buffer(
            "covariance_indices",
            torch.triu_indices(12, 12),
            persistent=False,
        )
        self.covariance_norm = nn.BatchNorm1d(78)
        self.covariance_head = nn.Linear(78, 10)
        nn.init.zeros_(self.covariance_head.weight)
        nn.init.zeros_(self.covariance_head.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
=======
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        projected = self.covariance_projection(
            gated_feature_map
        ).flatten(2)
        projected = projected - projected.mean(dim=2, keepdim=True)
        covariance = torch.bmm(
            projected, projected.transpose(1, 2)
        ) / projected.shape[-1]
        covariance = covariance[
            :,
            self.covariance_indices[0],
            self.covariance_indices[1],
        ]
        covariance_logits = self.covariance_head(
            self.covariance_norm(covariance)
        )
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + covariance_logits
        )
>>>>>>> REPLACE