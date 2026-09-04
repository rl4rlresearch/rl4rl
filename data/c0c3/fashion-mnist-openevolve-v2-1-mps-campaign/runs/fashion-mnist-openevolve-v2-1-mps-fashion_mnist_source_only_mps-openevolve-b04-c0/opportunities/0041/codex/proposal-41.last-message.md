MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.019287109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938580917.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.01904296875% to 3.019287109375%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: The current 3.01904296875% rate achieved 9,359 correct with the best cross-entropy, while 3.0234375% lost one prediction; bisecting toward the timed-out 3.01953125% trial is the closest conservative boundary refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301904296875)
=======
                    average.lerp_(buffer.detach(), 0.03019287109375)
>>>>>>> REPLACE