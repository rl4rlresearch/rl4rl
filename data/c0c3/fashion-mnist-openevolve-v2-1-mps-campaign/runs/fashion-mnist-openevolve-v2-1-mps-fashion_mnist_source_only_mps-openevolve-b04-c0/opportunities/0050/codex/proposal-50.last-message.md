MECHANISM: Local EMA calibration-rate bisection

HYPOTHESIS: A 3.01950836181640625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Decrease only the floating-buffer EMA update from 3.0195159912109375% to the midpoint above the previously worse 3.019500732421875% rate.

EVIDENCE: The current rate is the best tested point, while both the nearest tested lower and higher rates produced worse cross-entropy with the same correct count; bisecting the wider lower-side interval is the most informative remaining local refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.0301950836181640625)
>>>>>>> REPLACE