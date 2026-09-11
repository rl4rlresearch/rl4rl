MECHANISM: Accuracy-invariant logit temperature refinement

HYPOTHESIS: Increasing the final positive logit scale from 1.104 to 1.1045 will retain all 9,360 correct predictions while reducing validation cross-entropy below 0.18600157318115235.

INTENDED_EDIT: Raise only the evaluation-time logit multiplier, leaving training, TTA views, center weighting, normalization, and class rankings unchanged.

EVIDENCE: Center-weight refinement has plateaued at 9,360 correct with microscopic cross-entropy differences; a positive global logit scale preserves argmax predictions exactly while directly probing the remaining calibration objective.

<<<<<<< SEARCH
        return 1.104 * logits
=======
        return 1.1045 * logits
>>>>>>> REPLACE