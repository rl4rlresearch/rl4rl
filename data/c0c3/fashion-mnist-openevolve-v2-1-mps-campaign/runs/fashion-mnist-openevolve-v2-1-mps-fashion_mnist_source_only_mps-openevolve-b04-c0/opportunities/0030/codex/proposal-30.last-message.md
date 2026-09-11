MECHANISM: Fine-grained BatchNorm-buffer EMA boundary search

HYPOTHESIS: A 3.0625% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by improving responsiveness without crossing the accuracy boundary observed at 3.125%.

INTENDED_EDIT: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.0625%.

EVIDENCE: The 3% update achieved the best 9,359-correct result; 3.125% lowered cross-entropy but lost one correct prediction, so their midpoint is the most informative remaining boundary refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(buffer.detach(), 0.030625)
>>>>>>> REPLACE