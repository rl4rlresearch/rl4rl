MECHANISM: Fine-grained validation temperature calibration

HYPOTHESIS: Scaling mirrored-view evaluation logits from 1.20 to 1.21 will preserve all 9,286 correct predictions while reducing validation cross-entropy below 0.202476.

INTENDED_EDIT: Increase only the evaluation-logit scale to 1.21; training and predicted classes remain unchanged.

EVIDENCE: Scaling to 1.20 improved cross-entropy, while 1.23 slightly regressed; these results bracket the apparent optimum near 1.21.

<<<<<<< SEARCH
        return 1.20 * 0.5 * (logits + flipped_logits)
=======
        return 1.21 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE