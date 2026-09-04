MECHANISM: Quadratic-fit evaluation temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.184 will preserve all 9,330 argmax predictions and reduce validation cross-entropy below 0.196639.

INTENDED_EDIT: Change only the evaluation-time symmetric ensemble scale from 1.15 to 1.184.

EVIDENCE: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while cross-entropy fell from 0.200571 to 0.198018 to 0.196639; their diminishing improvements place the estimated calibration optimum near 1.184.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE