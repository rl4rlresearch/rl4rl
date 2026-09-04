MECHANISM: Nonlinear cross-channel statistics interaction

HYPOTHESIS: A zero-initialized 12-unit nonlinear branch over the normalized mean/std/max vector will exceed 9,345 correct predictions by learning cross-channel statistic interactions that the existing linear residual head and channelwise gate cannot represent.

INTENDED_EDIT: Add a compact 192→12→10 GELU residual branch using the existing normalized statistics, raising learned parameters from 247,546 to 249,992 without new spatial reductions.

EVIDENCE: The statistics bypass materially improved correctness, and full cross-channel recalibration reached 9,345 but was too slow; operating after the already-computed reductions tests cross-channel nonlinear context with negligible additional computation.

<<<<<<< SEARCH
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
=======
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.statistics_interaction = nn.Sequential(
            nn.Linear(64 * 3, 12),
            nn.GELU(),
            nn.Linear(12, 10),
        )
        nn.init.zeros_(self.statistics_interaction[-1].weight)
        nn.init.zeros_(self.statistics_interaction[-1].bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
=======
        normalized_statistics = self.statistics_norm(statistics)
        residual_logits = self.statistics_head(normalized_statistics)
        interaction_logits = self.statistics_interaction(
            normalized_statistics
        )
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + interaction_logits
        )
>>>>>>> REPLACE