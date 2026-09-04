MECHANISM: Convex local refinement of inference temperature

HYPOTHESIS: Scaling ensemble logits by 1.03592 will preserve all 9,243 predictions while reducing cross-entropy below 0.20883301086425782.

INTENDED_EDIT: Replace the 1.01 inference-only logit scale with the estimated local optimum, leaving training and ensembling unchanged.

EVIDENCE: Scales 1.0334 and 1.0360 achieved cross-entropies 0.20883404388427734 and 0.20883301086425782 with identical predictions; local convex interpolation places the minimum near 1.03592.

<<<<<<< SEARCH
            logits = 1.01 * (
=======
            logits = 1.03592 * (
>>>>>>> REPLACE