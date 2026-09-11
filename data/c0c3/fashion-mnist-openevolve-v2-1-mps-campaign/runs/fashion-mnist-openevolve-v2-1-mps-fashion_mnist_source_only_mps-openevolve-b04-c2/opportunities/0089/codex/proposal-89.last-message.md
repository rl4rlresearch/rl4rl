MECHANISM: Three-point quadratic logit recalibration

HYPOTHESIS: Evaluation scale 1.20485 will preserve all 9,281 argmax predictions while reducing validation cross-entropy below 0.20671812667846678.

INTENDED_EDIT: Change only the evaluation-time logit multiplier from 1.206 to the estimated calibration optimum 1.20485.

EVIDENCE: With label smoothing 0.023, scales 1.17817, 1.19, and 1.206 produced cross-entropies 0.2068297211, 0.2067525513, and 0.2067181267 respectively; quadratic interpolation estimates the minimum near 1.20485, and positive scaling cannot alter predicted classes.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.206 * logits
=======
        if not self.training:
            logits = 1.20485 * logits
>>>>>>> REPLACE