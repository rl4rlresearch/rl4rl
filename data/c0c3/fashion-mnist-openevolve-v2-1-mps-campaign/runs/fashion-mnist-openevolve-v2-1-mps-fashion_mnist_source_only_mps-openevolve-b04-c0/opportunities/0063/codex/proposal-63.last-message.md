MECHANISM: Lower-side EMA calibration-rate bisection

HYPOTHESIS: A 3.0195157825946807861328125% buffer EMA update will preserve 9,359 correct predictions while lower validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower, worse-performing rate.

EVIDENCE: The current 3.01951587200164794921875% rate achieved the best score, while 3.019515693187713623046875% preserved 9,359 correct but increased cross-entropy; their midpoint is the most informative untested lower-side refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.030195157825946807861328125)
>>>>>>> REPLACE