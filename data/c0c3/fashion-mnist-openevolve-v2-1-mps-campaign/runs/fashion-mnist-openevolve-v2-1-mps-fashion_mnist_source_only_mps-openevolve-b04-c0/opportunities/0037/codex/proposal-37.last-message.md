MECHANISM: Fine-grained BatchNorm-buffer EMA boundary bisection

HYPOTHESIS: A 3.01953125% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.1938593.

INTENDED_EDIT: Increase only the floating-buffer EMA update from 3.015625% to 3.01953125%, preserving the 4% parameter EMA and all other behavior.

EVIDENCE: The current 3.015625% rate achieved 9,359 correct, while 3.0234375% lowered cross-entropy but lost one prediction; their midpoint is the most informative untested boundary refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03015625)
=======
                    average.lerp_(buffer.detach(), 0.0301953125)
>>>>>>> REPLACE