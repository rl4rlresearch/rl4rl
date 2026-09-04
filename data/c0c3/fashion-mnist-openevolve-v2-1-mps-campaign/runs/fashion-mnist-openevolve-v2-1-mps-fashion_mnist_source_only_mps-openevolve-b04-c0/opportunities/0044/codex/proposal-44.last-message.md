MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.019500732421875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579308.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.01947021484375% to 3.019500732421875%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: Successive increases through 3.01947021484375% preserved 9,359 correct while monotonically lowering cross-entropy; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.0301947021484375)
=======
                    average.lerp_(buffer.detach(), 0.03019500732421875)
>>>>>>> REPLACE