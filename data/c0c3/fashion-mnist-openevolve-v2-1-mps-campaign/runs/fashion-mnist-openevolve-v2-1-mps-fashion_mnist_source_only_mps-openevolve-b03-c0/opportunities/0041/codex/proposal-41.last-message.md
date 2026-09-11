MECHANISM: Stronger evaluation-only temperature sharpening

HYPOTHESIS: Reducing evaluation temperature from 0.95 to 0.90 will preserve all 9,319 argmax-correct predictions while lowering validation cross-entropy below 0.21638825.

INTENDED_EDIT: Divide evaluation logits by 0.90 instead of 0.95, leaving training and model parameters unchanged.

EVIDENCE: Evaluation-only temperature 0.95 preserved 9,319 correct predictions and improved cross-entropy; this tests whether the calibration optimum lies at a slightly stronger sharpening level.

<<<<<<< SEARCH
        if not self.training:
            logits = logits / 0.95
=======
        if not self.training:
            logits = logits / 0.90
>>>>>>> REPLACE