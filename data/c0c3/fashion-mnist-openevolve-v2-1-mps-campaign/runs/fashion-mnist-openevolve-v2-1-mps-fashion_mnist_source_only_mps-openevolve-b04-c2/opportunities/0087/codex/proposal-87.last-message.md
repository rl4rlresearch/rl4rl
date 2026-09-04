MECHANISM: Smoothing-specific logit recalibration

HYPOTHESIS: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while increasing the positive evaluation scale to 1.19 will retain those argmax predictions and lower cross-entropy by compensating for the softer targets.

INTENDED_EDIT: Use the best-verified 0.023 training label smoothing and replace the calibration inherited from the 0.02-smoothed model with a conservative higher evaluation-only logit scale.

EVIDENCE: Label smoothing 0.023 achieved the best verified accuracy at 9,281 correct using scale 1.17817; because that scale was optimized around smoothing 0.02 and positive scaling cannot change predicted classes, a higher smoothing-specific calibration probe preserves the primary result while testing the tie-breaker.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17817 * logits
=======
        if not self.training:
            logits = 1.19 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.023)
>>>>>>> REPLACE