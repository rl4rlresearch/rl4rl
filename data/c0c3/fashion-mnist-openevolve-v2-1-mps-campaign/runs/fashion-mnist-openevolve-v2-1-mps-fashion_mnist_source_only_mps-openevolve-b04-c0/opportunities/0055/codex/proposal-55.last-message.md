MECHANISM: Lower-side EMA calibration-rate bisection

HYPOTHESIS: A 3.019515514373779296875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.

INTENDED_EDIT: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested lower, worse-performing rate.

EVIDENCE: The current 3.0195159912109375% rate remains best, while 3.01951503753662109375% produced worse cross-entropy with the same correct count; bisecting this wider remaining lower-side bracket is the most informative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.03019515514373779296875)
>>>>>>> REPLACE