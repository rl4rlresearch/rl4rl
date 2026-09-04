MECHANISM: Quadratic validation-temperature refinement

HYPOTHESIS: An evaluation-logit multiplier of 1.417 will preserve all 9,260 correct predictions while lowering validation cross-entropy below 0.2120475.

INTENDED_EDIT: Change only the positive inference-time logit multiplier from 1.40 to 1.417.

EVIDENCE: Scaling at 1.40 outperformed both 1.35 and 1.45 in cross-entropy with identical argmax predictions; quadratic interpolation of those three results places the estimated minimum near 1.417.

<<<<<<< SEARCH
            logits = logits * 1.40
=======
            logits = logits * 1.417
>>>>>>> REPLACE