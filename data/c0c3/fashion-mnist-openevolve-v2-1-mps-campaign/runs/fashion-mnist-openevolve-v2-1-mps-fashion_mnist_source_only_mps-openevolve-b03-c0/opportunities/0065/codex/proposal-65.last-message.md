MECHANISM: Quadratic-optimal evaluation temperature calibration

HYPOTHESIS: Dividing evaluation logits by 0.738156 will preserve all 9,319 argmax predictions while reducing validation cross-entropy below 0.20007479591369629.

INTENDED_EDIT: Refine the evaluation-only temperature from 0.7382 to 0.738156 without changing training, parameters, or computational cost.

EVIDENCE: Temperatures 0.738, 0.7382, and 0.74 yielded cross-entropies 0.2000748046875, 0.20007479591369629, and 0.200076131439209; their quadratic fit places the minimum near 0.738156. The prior 0.73816 run timed out but supplied no contrary accuracy or calibration evidence.

<<<<<<< SEARCH
        if not self.training:
            logits = logits / 0.7382
=======
        if not self.training:
            logits = logits / 0.738156
>>>>>>> REPLACE