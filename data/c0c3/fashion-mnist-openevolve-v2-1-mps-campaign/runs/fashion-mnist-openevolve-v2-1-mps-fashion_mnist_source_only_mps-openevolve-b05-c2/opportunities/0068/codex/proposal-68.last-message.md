MECHANISM: Proven float32 temperature calibration restoration

HYPOTHESIS: The best-verified sharpening coefficient will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22237594985961914 to approximately 0.22237433319091796.

INTENDED_EDIT: Replace the current 76.74%-denominator calibration with Reference Design 1’s explicit float32 coefficient.

EVIDENCE: Reference Design 1 achieved the highest observed validation_score, 9192.40904000225; the next larger and immediately smaller float32 coefficients were both worse.

<<<<<<< SEARCH
        return normalized_logits * (1400.0 / 1279.0)
=======
        return normalized_logits * 1.0978671312332153
>>>>>>> REPLACE