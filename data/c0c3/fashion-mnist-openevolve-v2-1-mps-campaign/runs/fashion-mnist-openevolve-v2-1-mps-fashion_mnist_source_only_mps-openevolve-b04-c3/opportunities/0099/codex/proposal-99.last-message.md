MECHANISM: Locally refined inference-temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.03577 will preserve all 9,243 argmax predictions while reducing validation cross-entropy below 0.20883300514221192.

INTENDED_EDIT: Apply a 1.03577 positive scale to inference-only flip-ensemble logits, leaving training and ensembling unchanged.

EVIDENCE: Scale 1.03592 achieved the best cross-entropy, while 1.0360 was slightly worse and 1.0334 was materially worse; quadratic interpolation of these verified measurements places the local minimum near 1.03577.

<<<<<<< SEARCH
            logits = (
                torch.logaddexp(
=======
            logits = 1.03577 * (
                torch.logaddexp(
>>>>>>> REPLACE