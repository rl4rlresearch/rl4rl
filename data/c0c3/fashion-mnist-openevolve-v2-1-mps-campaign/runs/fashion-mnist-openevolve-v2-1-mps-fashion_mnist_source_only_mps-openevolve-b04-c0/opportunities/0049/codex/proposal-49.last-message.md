MECHANISM: Local EMA calibration-rate bisection

HYPOTHESIS: A 3.01951694488525390625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.0195159912109375% to the midpoint before the worse-performing 3.0195178985595703125% trial.

EVIDENCE: The current rate produced the best cross-entropy, while both tested higher rates worsened it with the same correct count; bisecting the nearest bracket is the most informative local refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.03019516944885254)
>>>>>>> REPLACE