MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.01947021484375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579536.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.0194091796875% to 3.01947021484375%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: Successive increases through 3.0194091796875% preserved 9,359 correct while monotonically lowering cross-entropy; 3.0234375% lost one prediction, so the midpoint toward the timed-out 3.01953125% trial is the closest conservative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030194091796875)
=======
                    average.lerp_(buffer.detach(), 0.0301947021484375)
>>>>>>> REPLACE