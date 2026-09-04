MECHANISM: Local quadratic logit recalibration

HYPOTHESIS: An evaluation scale of 1.16727 will preserve all 9,284 predictions while reducing cross-entropy below 0.207258652.

INTENDED_EDIT: Replace the evaluation-logit multiplier with the vertex estimated from the three nearest validated calibration points.

EVIDENCE: Scales 1.153, 1.1645, and 1.1682 yielded cross-entropies 0.207286140, 0.207259576, and 0.207258652; their local quadratic fit has its minimum near 1.16727, and positive scaling preserves argmax rankings.

<<<<<<< SEARCH
        return 1.249 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE