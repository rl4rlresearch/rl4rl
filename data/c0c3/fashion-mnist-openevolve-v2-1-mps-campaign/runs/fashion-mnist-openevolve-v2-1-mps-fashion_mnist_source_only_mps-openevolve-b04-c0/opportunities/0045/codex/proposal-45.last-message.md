MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.0195159912109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579266.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.019500732421875% to 3.0195159912109375%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: Successive increases through 3.019500732421875% preserved 9,359 correct while monotonically lowering cross-entropy; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03019500732421875)
=======
                    average.lerp_(buffer.detach(), 0.030195159912109375)
>>>>>>> REPLACE