MECHANISM: Fine-grained BatchNorm-buffer EMA boundary refinement

HYPOTHESIS: A 3.03125% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by moving toward the better-calibrated 3.0625% rate without crossing its observed accuracy boundary.

INTENDED_EDIT: Keep the parameter EMA update at 4% and increase only the floating-buffer EMA update from 3% to 3.03125%.

EVIDENCE: The 3% buffer update achieved the best score with 9,359 correct, while 3.0625% lowered cross-entropy to 0.193843 but lost one correct prediction; their midpoint is the closest untested boundary refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(buffer.detach(), 0.0303125)
>>>>>>> REPLACE