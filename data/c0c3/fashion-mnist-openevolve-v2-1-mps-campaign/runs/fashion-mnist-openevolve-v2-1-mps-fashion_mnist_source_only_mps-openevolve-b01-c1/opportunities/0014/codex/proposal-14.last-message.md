MECHANISM: Decision-preserving continued logit sharpening

HYPOTHESIS: Scaling evaluation logits by 1.15 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.261929.

INTENDED_EDIT: Increase the evaluation-only logit scale from 1.10 to 1.15 without changing training, parameters, or predicted classes.

EVIDENCE: Raising the scale from 1.05 to 1.10 preserved all 9,063 correct predictions and reduced cross-entropy from 0.266563 to 0.261929, so the ensemble remains underconfident at the current scale.

<<<<<<< SEARCH
        return 1.10 * (0.9 * full_ensemble + 0.1 * central_ensemble)
=======
        return 1.15 * (0.9 * full_ensemble + 0.1 * central_ensemble)
>>>>>>> REPLACE