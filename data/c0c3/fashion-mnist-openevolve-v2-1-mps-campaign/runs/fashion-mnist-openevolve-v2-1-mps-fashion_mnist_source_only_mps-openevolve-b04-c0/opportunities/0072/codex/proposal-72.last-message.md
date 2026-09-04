MECHANISM: Upper-side EMA calibration plateau bisection

HYPOTHESIS: A 3.019515874795615673065185546875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher rate that tied it.

EVIDENCE: The current 3.01951587200164794921875% rate and 3.01951587758958339691162109375% both achieved 9,359 correct with cross-entropy 0.1938579158782959; their untested midpoint most precisely probes the remaining upper-side plateau.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.03019515874795615673065185546875)
>>>>>>> REPLACE