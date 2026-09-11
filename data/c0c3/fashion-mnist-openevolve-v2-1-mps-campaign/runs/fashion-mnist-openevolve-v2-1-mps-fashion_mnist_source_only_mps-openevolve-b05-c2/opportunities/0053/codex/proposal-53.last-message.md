MECHANISM: Order-preserving quadratic-fit temperature calibration

HYPOTHESIS: Retrying the effective 76.74% denominator will retain 9,192 correct predictions while reducing validation cross-entropy below 0.2223825225830078.

INTENDED_EDIT: Keep the proven 84%-denominator logits and max-centering, but change the sharpening factor from 84/76 to 84/76.74.

EVIDENCE: Cross-entropy at effective denominators of 68%, 76%, and 84% places the fitted minimum near 76.74%; its prior verification timed out, so it remains the most informative unresolved calibration probe.

<<<<<<< SEARCH
        return normalized_logits * (21.0 / 19.0)
=======
        return normalized_logits * (1400.0 / 1279.0)
>>>>>>> REPLACE