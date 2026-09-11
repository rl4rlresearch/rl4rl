MECHANISM: Learned projected channel-covariance residual head

HYPOTHESIS: A zero-initialized covariance head will exceed 9,360 correct predictions by exposing second-order channel co-activation evidence unavailable to the existing per-channel statistics and 30-unit spatial bottleneck.

INTENDED_EDIT: Project gated feature maps to 15 channels, compute their 120 unique spatial covariance terms, normalize them, and add learned residual logits. This raises the model to 249,956 parameters.

EVIDENCE: Repeated center-weight refinements plateaued at 9,360 correct with only microscopic cross-entropy gains, indicating inference aggregation is saturated. The old assumption is that marginal channel statistics are sufficient; this patch instead models joint channel interactions with little additional computation.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.interaction_projection = nn.Conv2d(
            64, 15, kernel_size=1, bias=False
        )
        interaction_indices = torch.triu_indices(15, 15)
        self.register_buffer(
            "interaction_rows", interaction_indices[0], persistent=False
        )
        self.register_buffer(
            "interaction_columns", interaction_indices[1], persistent=False
        )
        self.interaction_norm = nn.BatchNorm1d(120)
        self.interaction_head = nn.Linear(120, 10)
        nn.init.zeros_(self.interaction_head.weight)
        nn.init.zeros_(self.interaction_head.bias)
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
        interaction_map = self.interaction_projection(
            gated_feature_map
        ).flatten(2)
        interaction_map = interaction_map - interaction_map.mean(
            dim=2, keepdim=True
        )
        interaction_covariance = torch.bmm(
            interaction_map, interaction_map.transpose(1, 2)
        ) / interaction_map.shape[2]
        interaction_statistics = interaction_covariance[
            :, self.interaction_rows, self.interaction_columns
        ]
        interaction_logits = self.interaction_head(
            self.interaction_norm(interaction_statistics)
        )
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + interaction_logits
        )
>>>>>>> REPLACE