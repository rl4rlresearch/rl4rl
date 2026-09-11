MECHANISM: Upper-side EMA calibration plateau bisection

HYPOTHESIS: A 3.01951587758958339691162109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher rate that tied it.

EVIDENCE: The current 3.01951587200164794921875% rate and 3.0195158831775188446044921875% both achieved 9,359 correct with cross-entropy 0.1938579158782959; their untested midpoint is the most informative remaining refinement near the upper edge of the observed optimum.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.0301951587758958339691162109375)
>>>>>>> REPLACE