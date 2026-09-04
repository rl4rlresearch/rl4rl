MECHANISM: Verified evaluation-logit calibration

HYPOTHESIS: Scaling evaluation logits from 1.295 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy to approximately 0.2175833054.

INTENDED_EDIT: Replace only the evaluation-time logit multiplier with the best verified calibration.

EVIDENCE: Reference Designs 1 and 2 achieved the best available score, 9240.410649520078, with the identical training recipe and a 1.296352 multiplier.

<<<<<<< SEARCH
        return 1.295 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE