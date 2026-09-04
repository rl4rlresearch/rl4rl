MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.0194091796875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938580013.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.019287109375% to 3.0194091796875%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: The current 3.019287109375% rate achieved 9,359 correct with the best cross-entropy, while 3.0234375% lost one prediction; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03019287109375)
=======
                    average.lerp_(buffer.detach(), 0.030194091796875)
>>>>>>> REPLACE