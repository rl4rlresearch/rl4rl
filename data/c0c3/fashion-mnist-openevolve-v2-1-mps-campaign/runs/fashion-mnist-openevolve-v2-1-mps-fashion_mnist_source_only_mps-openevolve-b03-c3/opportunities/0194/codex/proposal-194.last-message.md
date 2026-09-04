MECHANISM: Best-validated global logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.16727 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207258542.

INTENDED_EDIT: Replace only the evaluation-logit multiplier with the strongest verified scale.

EVIDENCE: Reference Design 2 achieved the highest available validation_score at scale 1.16727 with 9,284 correct and cross-entropy 0.207258541; the current 1.167286 scale produced slightly higher cross-entropy, while positive global scaling cannot change class rankings.

<<<<<<< SEARCH
        return 1.167286 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE