MECHANISM: Upper-side EMA calibration quarter-step

HYPOTHESIS: A 3.0195159614086151123046875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the twice-timed-out upper midpoint.

EVIDENCE: The current 3.01951587200164794921875% rate is best, while the nearest completed higher rate was worse; because the direct upper midpoint timed out twice without performance evidence, this untested quarter-step probes the unresolved upper interval.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.030195159614086151123046875)
>>>>>>> REPLACE