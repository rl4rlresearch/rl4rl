MECHANISM: Conservative BatchNorm-buffer EMA boundary refinement

HYPOTHESIS: A 3.0234375% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.1938593 by moving toward the better-calibrated higher-rate region without reaching the observed accuracy-loss boundary.

INTENDED_EDIT: Keep the parameter EMA update at 4% and increase only the floating-buffer EMA update from 3.015625% to 3.0234375%.

EVIDENCE: Raising the buffer update from 3% to 3.015625% preserved 9,359 correct and improved cross-entropy from approximately 0.193865 to 0.1938593, while 3.0625% lost one prediction; bisecting toward the uncompleted 3.03125% trial is the closest informative refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03015625)
=======
                    average.lerp_(buffer.detach(), 0.030234375)
>>>>>>> REPLACE