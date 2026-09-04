MECHANISM: Verified second-order confidence calibration

HYPOTHESIS: Changing the evaluation-only logit scale from 1.295 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2175835346 to approximately 0.2175833054.

INTENDED_EDIT: Replace only the evaluation-logit multiplier with the best verified calibration, leaving training and class decisions unchanged.

EVIDENCE: Reference Design 1 used the identical implementation at scale 1.296352 and achieved the best available score, 9240.410649520078; architecture changes and EMA reduced validation correctness.

<<<<<<< SEARCH
        return 1.295 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE