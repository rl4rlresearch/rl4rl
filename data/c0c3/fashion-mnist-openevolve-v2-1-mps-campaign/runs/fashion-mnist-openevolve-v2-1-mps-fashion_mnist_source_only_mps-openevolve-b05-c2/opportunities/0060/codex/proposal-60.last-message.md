MECHANISM: Proven order-preserving temperature calibration

HYPOTHESIS: Restoring the verified 76.512% effective denominator will preserve 9,192 correct predictions and reduce validation cross-entropy below the current 0.22237594985961914.

INTENDED_EDIT: Replace the current 76.74% evaluation-logit denominator with the best verified 76.512% calibration while retaining max-centering and class ordering.

EVIDENCE: Reference Design 3 achieved 9,192 correct with the best observed cross-entropy, 0.22237433319091796; finer denominator probes were worse, and single-rounding variants repeatedly timed out.

<<<<<<< SEARCH
        return normalized_logits * (1400.0 / 1279.0)
=======
        return normalized_logits * (10500.0 / 9564.0)
>>>>>>> REPLACE