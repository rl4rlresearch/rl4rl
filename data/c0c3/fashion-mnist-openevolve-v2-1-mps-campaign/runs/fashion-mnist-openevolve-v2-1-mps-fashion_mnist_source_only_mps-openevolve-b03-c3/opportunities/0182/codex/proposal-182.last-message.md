MECHANISM: Three-point quadratic logit calibration

HYPOTHESIS: An evaluation scale of 1.1682 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207259576.

INTENDED_EDIT: Replace the evaluation-logit multiplier with the fitted cross-entropy minimum, leaving training, TTA weights, and class rankings unchanged.

EVIDENCE: Scales 1.153, 1.1645, and 1.248 produced cross-entropies 0.207286140, 0.207259576, and 0.208037965 respectively; a quadratic fit places the minimum near 1.1682, and positive global scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.248 * logits
=======
        return 1.1682 * logits
>>>>>>> REPLACE