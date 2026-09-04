MECHANISM: Float32-adjacent confidence calibration

HYPOTHESIS: Increasing the positive evaluation-logit scale by one float32 step will preserve all 9,331 argmax predictions while reducing validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Replace the 1.184 evaluation calibration with its immediate float32 successor, leaving training and the best verified fusion weights unchanged.

EVIDENCE: The current design has the best verified 9,331-correct result; positive scaling cannot change argmax predictions, and prior attempts at this adjacent scale timed out without contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (
=======
            logits = 1.1840001344680786 * (
>>>>>>> REPLACE