MECHANISM: Upper-side EMA calibration quarter-step

HYPOTHESIS: A 3.0195158831775188446044921875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.

INTENDED_EDIT: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the most recent timed-out upper midpoint.

EVIDENCE: The current 3.01951587200164794921875% rate is best; the nearest completed higher rates worsened cross-entropy, while 3.019515894353389739990234375% timed out without performance evidence, motivating a smaller untested upper step.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)
=======
                    average.lerp_(buffer.detach(), 0.030195158831775188446044921875)
>>>>>>> REPLACE