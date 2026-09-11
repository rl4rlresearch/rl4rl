MECHANISM: Conservative BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.0185546875% buffer EMA update will preserve 9,359 correct predictions while lowering cross-entropy below 0.1938586182.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.017578125% to 3.0185546875%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: The current 3.017578125% rate preserved 9,359 correct with the best cross-entropy; 3.0234375% lowered cross-entropy but lost one prediction, while 3.01953125% timed out. Bisecting the remaining conservative interval is the closest informative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03017578125)
=======
                    average.lerp_(buffer.detach(), 0.030185546875)
>>>>>>> REPLACE