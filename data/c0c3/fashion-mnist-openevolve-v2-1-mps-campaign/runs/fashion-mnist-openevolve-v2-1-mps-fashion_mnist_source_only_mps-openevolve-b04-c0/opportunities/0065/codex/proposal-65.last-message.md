MECHANISM: Upper-side EMA calibration-rate bisection retry

HYPOTHESIS: A 3.019516050815582275390625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested higher, worse-performing rate.

EVIDENCE: The current 3.01951587200164794921875% rate is best, while 3.0195162296295166015625% was slightly worse; their midpoint remains the most informative candidate because its previous verification timed out without producing performance evidence.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.03019516050815582275390625)
>>>>>>> REPLACE