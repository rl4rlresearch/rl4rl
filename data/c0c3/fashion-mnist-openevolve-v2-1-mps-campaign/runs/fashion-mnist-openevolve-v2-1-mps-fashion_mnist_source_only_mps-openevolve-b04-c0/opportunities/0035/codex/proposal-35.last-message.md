MECHANISM: Conservative BatchNorm-buffer EMA boundary refinement

HYPOTHESIS: A 3.015625% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by moving slightly toward the better-calibrated 3.0625% rate without crossing its observed accuracy boundary.

INTENDED_EDIT: Keep the parameter EMA update at 4% and increase only the floating-buffer EMA update from 3% to 3.015625%.

EVIDENCE: The 3% update achieved the best score with 9,359 correct, while 3.0625% lowered cross-entropy to 0.193843 but lost one prediction; after the 3.03125% midpoint timed out, 3.015625% is the closest conservative refinement between the verified endpoints.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(buffer.detach(), 0.03015625)
>>>>>>> REPLACE