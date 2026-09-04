MECHANISM: Best-validated global logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.16727 will preserve all 9,284 argmax predictions while lowering cross-entropy below the current 0.207259576.

INTENDED_EDIT: Replace only the evaluation-logit multiplier with the strongest verified calibration.

EVIDENCE: Reference Design 1 achieved the highest available validation_score, 9284.414161493147, with 9,284 correct and cross-entropy 0.207258541; positive scaling cannot change argmax predictions.

<<<<<<< SEARCH
        return 1.1645 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE