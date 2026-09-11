MECHANISM: Best-validated global logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.16727 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258652.

INTENDED_EDIT: Replace only the evaluation-logit multiplier with the strongest verified scale.

EVIDENCE: Reference Design 2 achieved the highest reported validation_score, 9284.414161493147, with 9,284 correct and cross-entropy 0.207258541; positive global scaling cannot change argmax predictions.

<<<<<<< SEARCH
        return 1.1682 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE