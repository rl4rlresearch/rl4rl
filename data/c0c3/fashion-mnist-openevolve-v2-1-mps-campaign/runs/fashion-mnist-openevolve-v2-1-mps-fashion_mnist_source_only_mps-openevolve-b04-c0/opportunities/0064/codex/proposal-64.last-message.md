MECHANISM: Lower-side EMA calibration bisection

HYPOTHESIS: A 3.01951582729816436767578125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower rate that tied it.

EVIDENCE: The current 3.01951587200164794921875% rate and 3.0195157825946807861328125% both achieved 9,359 correct with cross-entropy 0.1938579158782959, while the next-lower tested rate was worse; bisecting the remaining plateau is the most informative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.0301951582729816436767578125)
>>>>>>> REPLACE