MECHANISM: Upper-side EMA calibration-rate bisection

HYPOTHESIS: A 3.019515894353389739990234375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest completed higher, worse-performing rate.

EVIDENCE: The current 3.01951587200164794921875% rate is best, while 3.01951591670513153076171875% preserved 9,359 correct but increased cross-entropy; their untested midpoint most directly refines the remaining upper interval.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.03019515894353389739990234375)
>>>>>>> REPLACE