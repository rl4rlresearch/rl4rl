MECHANISM: Fine-grained convex temperature calibration

HYPOTHESIS: Scaling the ensemble logits by 1.356 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.214452343.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.355 to 1.356.

EVIDENCE: Cross-entropy fell from 0.214455883 at 1.35 to 0.214452343 at 1.355; local finite-difference curvature estimates place the minimum near 1.356, while positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.355 * aggregate_logits
=======
        return 1.356 * aggregate_logits
>>>>>>> REPLACE