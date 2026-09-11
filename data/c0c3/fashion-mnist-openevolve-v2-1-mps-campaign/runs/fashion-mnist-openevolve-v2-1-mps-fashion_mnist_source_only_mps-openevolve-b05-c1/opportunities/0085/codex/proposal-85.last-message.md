MECHANISM: Post-normalization statistics recalibration

HYPOTHESIS: Moving raw-mean channel gating after statistics BatchNorm will exceed 9,345 correct predictions by preserving recalibration while preventing sample-dependent gates from altering BatchNorm statistics.

INTENDED_EDIT: Normalize the unchanged mean/std/max descriptors before applying the existing channel gate; architecture, parameters, optimizer, and runtime remain effectively unchanged.

EVIDENCE: Raw-mean diagonal gating achieved the best 9,345-correct result, while normalized gate conditioning regressed to 9,318 and removing statistics gating was inconclusive due to timeout; this isolates gate placement without changing its successful conditioning signal.

<<<<<<< SEARCH
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
=======
        statistics = self.statistics_norm(statistics)
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(statistics)
>>>>>>> REPLACE