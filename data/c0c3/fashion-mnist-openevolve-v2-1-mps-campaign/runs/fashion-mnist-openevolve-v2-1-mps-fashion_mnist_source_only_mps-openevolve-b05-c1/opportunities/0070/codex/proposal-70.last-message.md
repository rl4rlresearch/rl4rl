MECHANISM: Class-coherent spatial evidence pooling

HYPOTHESIS: Mixing channels into class-specific evidence maps before smooth-max spatial pooling will exceed 9,345 correct predictions by detecting co-located class features that channelwise statistics and the rank-30 flattened bottleneck cannot represent directly.

INTENDED_EDIT: Add a zero-initialized 1×1 class-evidence head over the recalibrated feature map, aggregate each class map with normalized log-sum-exp pooling, and add those scores to the existing logits.

EVIDENCE: The channelwise statistics bypass and diagonal gate improved the best result to 9,345, but further global-statistic conditioning failed or timed out. This challenges their shared pool-before-class-mixing assumption with a cheap class-mixing-before-pooling path while preserving the current model exactly at initialization.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.evidence_head = nn.Conv2d(64, 10, kernel_size=1)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
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
        evidence_maps = self.evidence_head(gated_feature_map)
        evidence_logits = torch.logsumexp(
            evidence_maps.flatten(2), dim=2
        ) - math.log(49.0)
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + evidence_logits
        )
>>>>>>> REPLACE