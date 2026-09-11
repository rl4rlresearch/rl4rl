MECHANISM: Identity-initialized diagonal quadratic statistics enrichment

HYPOTHESIS: Adding learnable curvature to each normalized statistic will exceed 9,345 correct predictions while remaining within the time limit by capturing nonlinear distribution-shape evidence without an additional classifier branch.

INTENDED_EDIT: Add 192 learned curvature coefficients that augment each normalized mean/std/max feature with a centered quadratic term before the existing statistics head, increasing parameters to 247,738.

EVIDENCE: The diagonal channel gate produced the best result of 9,345 correct, while the 12-unit nonlinear statistics branch timed out; a factorized quadratic transform tests nonlinear statistics at substantially lower computational cost.

<<<<<<< SEARCH
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.statistics_curvature = nn.Parameter(torch.zeros(64 * 3))
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
=======
        normalized_statistics = self.statistics_norm(statistics)
        normalized_statistics = normalized_statistics + (
            self.statistics_curvature
            * (normalized_statistics.square() - 1.0)
        )
        residual_logits = self.statistics_head(normalized_statistics)
        return self.classifier(gated_feature_map) + residual_logits
>>>>>>> REPLACE