MECHANISM: Class-conditioned spatial attention residual

HYPOTHESIS: A lightweight class-specific attention head will exceed 9,360 correct predictions by learning where each class’s discriminative features occur, while remaining substantially cheaper than the timed-out covariance head.

INTENDED_EDIT: Add zero-output-initialized per-class evidence maps and learned per-class spatial attention, then aggregate localized evidence into residual logits.

EVIDENCE: Center-weight and temperature calibration plateaued at 9,360 correct, showing that inference calibration is saturated. The covariance experiment targeted richer interactions but timed out; this instead challenges the shared assumption that fixed flattening and class-agnostic global statistics adequately preserve spatial evidence.

<<<<<<< SEARCH
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.spatial_attention = nn.Conv2d(64, 10, kernel_size=1)
        self.spatial_evidence = nn.Conv2d(64, 10, kernel_size=1)
        nn.init.zeros_(self.spatial_evidence.weight)
        nn.init.zeros_(self.spatial_evidence.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
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
        attention = self.spatial_attention(
            gated_feature_map
        ).flatten(2).softmax(dim=-1)
        evidence = self.spatial_evidence(gated_feature_map).flatten(2)
        localized_logits = (attention * evidence).sum(dim=-1)
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + localized_logits
        )
>>>>>>> REPLACE