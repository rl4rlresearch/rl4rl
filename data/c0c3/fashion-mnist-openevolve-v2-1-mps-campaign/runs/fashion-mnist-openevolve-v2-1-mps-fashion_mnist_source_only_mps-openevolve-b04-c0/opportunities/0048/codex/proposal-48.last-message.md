MECHANISM: Local EMA calibration-rate bisection

HYPOTHESIS: A 3.0195178985595703125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.0195159912109375% to the midpoint before the worse-performing 3.019519805908203125% trial.

EVIDENCE: Cross-entropy improved monotonically through 3.0195159912109375%, but worsened at 3.019519805908203125% with the same correct count; bisecting this newly bracketed interval is the most informative local refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.030195178985595703125)
>>>>>>> REPLACE