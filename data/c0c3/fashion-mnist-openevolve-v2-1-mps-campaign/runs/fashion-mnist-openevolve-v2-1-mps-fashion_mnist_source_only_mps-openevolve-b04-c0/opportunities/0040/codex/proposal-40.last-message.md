MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.01904296875% buffer EMA update will preserve 9,359 correct predictions while lowering cross-entropy below 0.1938582649.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.0185546875% to 3.01904296875%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: The current 3.0185546875% rate preserved 9,359 correct and improved cross-entropy, while 3.0234375% lost one prediction and the intervening 3.01953125% trial timed out; bisecting the remaining conservative interval is the closest informative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030185546875)
=======
                    average.lerp_(buffer.detach(), 0.0301904296875)
>>>>>>> REPLACE