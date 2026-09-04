MECHANISM: Upper-side EMA calibration-rate bisection

HYPOTHESIS: A 3.0195162296295166015625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested higher, worse-performing rate.

EVIDENCE: The current 3.0195159912109375% rate remains best, while 3.019516468048095703125% produced slightly worse cross-entropy with the same correct count; bisecting this upper-side bracket is the most informative remaining refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.030195162296295166015625)
>>>>>>> REPLACE