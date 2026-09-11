MECHANISM: Lower-side EMA calibration-rate bisection

HYPOTHESIS: A 3.019512176513671875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Decrease only the floating-buffer EMA update from 3.0195159912109375% to the midpoint above the worse-performing 3.01950836181640625% trial.

EVIDENCE: The current rate remains the best tested point, while the nearest lower rate produced worse cross-entropy with the same correct count; bisecting that bracket is the most informative remaining local refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.03019512176513671875)
>>>>>>> REPLACE