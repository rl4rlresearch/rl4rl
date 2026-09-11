MECHANISM: Upper-side EMA calibration-rate bisection

HYPOTHESIS: A 3.01951634883880615234375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791625976562.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher, worse-performing rate.

EVIDENCE: The current 3.0195162296295166015625% rate achieved the best score, while 3.019516468048095703125% preserved 9,359 correct but slightly worsened cross-entropy; their midpoint is the most informative untested upper-side refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195162296295166015625)
=======
                    average.lerp_(buffer.detach(), 0.0301951634883880615234375)
>>>>>>> REPLACE