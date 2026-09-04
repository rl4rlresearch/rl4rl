MECHANISM: Lower-side EMA calibration plateau bisection

HYPOTHESIS: A 3.019515849649906158447265625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower rate that tied it.

EVIDENCE: Rates of 3.01951587200164794921875% and 3.01951582729816436767578125% both achieved 9,359 correct with cross-entropy 0.1938579158782959; their untested midpoint is the most informative refinement within the observed plateau.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.03019515849649906158447265625)
>>>>>>> REPLACE