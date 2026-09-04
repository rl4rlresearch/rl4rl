MECHANISM: Three-point convex temperature refinement

HYPOTHESIS: An evaluation-time logit scale of 1.0671952 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.18585700874328615.

INTENDED_EDIT: Replace only the positive evaluation-time logit multiplier with the quadratic minimum estimated from three completed measurements.

EVIDENCE: Scale 1.0664346 improved cross-entropy to 0.18585700874328615; the higher losses at 1.0658182 and 1.1035 place the three-point interpolated minimum near 1.0671952, while positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.0664346 * logits
=======
        return 1.0671952 * logits
>>>>>>> REPLACE