MECHANISM: Verified evaluation-logit calibration

HYPOTHESIS: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.

INTENDED_EDIT: Replace only the evaluation-time logit multiplier, leaving training and class decisions unchanged.

EVIDENCE: Reference Design 1 used the otherwise identical implementation at scale 1.296352 and achieved the best available score, 9240.410649520078; prior architecture and EMA experiments reduced validation correctness.

<<<<<<< SEARCH
        return 1.25 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE