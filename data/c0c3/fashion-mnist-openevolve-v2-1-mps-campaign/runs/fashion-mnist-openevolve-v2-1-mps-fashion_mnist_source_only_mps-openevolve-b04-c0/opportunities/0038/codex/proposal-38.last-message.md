MECHANISM: Conservative BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.017578125% buffer EMA update will preserve 9,359 correct predictions while lowering cross-entropy below 0.1938592953.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.015625% to 3.017578125%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: The current 3.015625% rate achieved 9,359 correct, while 3.0234375% lowered cross-entropy but lost one prediction; after the intervening 3.01953125% trial timed out, this conservative midpoint is the closest untested refinement on the accuracy-preserving side.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03015625)
=======
                    average.lerp_(buffer.detach(), 0.03017578125)
>>>>>>> REPLACE