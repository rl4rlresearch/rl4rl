MECHANISM: Quadratically estimated argmax-preserving temperature hardening

HYPOTHESIS: Using the best verified power-mean order and scaling its logits by 1.0334 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20894218406677245.

INTENDED_EDIT: Set the power-mean order to 0.5831695556640625 and apply a 1.0334 inference-only logit scale.

EVIDENCE: Scaling from 0.99 to 1.0 to 1.01 progressively reduced cross-entropy from 0.20918512001037598 to 0.20904547996520997 to 0.20894218406677245 without changing the 9,243 correct predictions; quadratic interpolation estimates the calibration optimum near 1.0334.

<<<<<<< SEARCH
            power = 0.583172607421875
            logits = (
=======
            power = 0.5831695556640625
            logits = 1.0334 * (
>>>>>>> REPLACE