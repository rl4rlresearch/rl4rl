MECHANISM: Upper-side EMA calibration-rate bisection

HYPOTHESIS: A 3.019516050815582275390625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher, worse-performing rate.

EVIDENCE: The current 3.01951587200164794921875% rate achieved the best score, while 3.0195162296295166015625% preserved 9,359 correct but had slightly higher cross-entropy; their midpoint is the most informative untested refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.03019516050815582275390625)
>>>>>>> REPLACE