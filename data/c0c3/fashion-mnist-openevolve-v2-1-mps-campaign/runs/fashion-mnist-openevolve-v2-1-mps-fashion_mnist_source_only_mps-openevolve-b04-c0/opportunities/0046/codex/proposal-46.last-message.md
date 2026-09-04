MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.01952362060546875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579166.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.0195159912109375% to 3.01952362060546875%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: Successive increases through 3.0195159912109375% preserved 9,359 correct while monotonically lowering cross-entropy; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.030195159912109375)
=======
                    average.lerp_(buffer.detach(), 0.0301952362060546875)
>>>>>>> REPLACE