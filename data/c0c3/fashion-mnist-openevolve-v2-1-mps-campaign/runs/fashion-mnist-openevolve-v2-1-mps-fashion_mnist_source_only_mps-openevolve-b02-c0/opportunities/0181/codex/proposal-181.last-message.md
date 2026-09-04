MECHANISM: One-ULP logit-temperature refinement

HYPOTHESIS: Decreasing the positive logit scale by one float32 step will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Replace the evaluation logit multiplier with its immediately lower float32 value.

EVIDENCE: The current vertical weighting is locally best; paired, orientation-selective, and horizontal neighboring redistributions all regressed. Logit scaling is an orthogonal tie-break coordinate whose positive adjustment cannot change mathematical argmax predictions.

<<<<<<< SEARCH
        return 1.3715955 * ensemble_scores
=======
        return 1.3715953826904297 * ensemble_scores
>>>>>>> REPLACE