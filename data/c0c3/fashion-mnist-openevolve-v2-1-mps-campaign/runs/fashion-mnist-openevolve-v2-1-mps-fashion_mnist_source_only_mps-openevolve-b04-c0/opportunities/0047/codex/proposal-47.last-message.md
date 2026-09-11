MECHANISM: Local EMA calibration-rate bisection

HYPOTHESIS: A 3.019519805908203125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.0195159912109375% to the midpoint before the tested 3.01952362060546875% endpoint, preserving all other behavior.

EVIDENCE: The current rate achieved the best completed cross-entropy, while 3.01952362060546875% retained 9,359 correct but produced slightly worse cross-entropy and exceeded the time limit; their midpoint is the most informative refinement of the newly bracketed optimum.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.03019519805908203125)
>>>>>>> REPLACE