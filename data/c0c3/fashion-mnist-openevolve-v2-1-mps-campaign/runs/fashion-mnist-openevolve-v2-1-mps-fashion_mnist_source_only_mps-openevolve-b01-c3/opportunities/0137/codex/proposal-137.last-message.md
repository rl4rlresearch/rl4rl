MECHANISM: Local quadratic refinement of agreement-conditioned calibration

HYPOTHESIS: A 7.7757513% disagreement adjustment will preserve all 9,287 pooled predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935852050782.

INTENDED_EDIT: Move the agreement-conditioning strength from 7.788524% to the unresolved quadratic-minimum estimate of 7.7757513%, leaving training and pooled logits unchanged.

EVIDENCE: Verified strengths of 7.708%, 7.727%, and 7.788524% progressively reduced cross-entropy while retaining 9,287 correct; quadratic interpolation estimated 7.7757513%, and its only verification timed out without subject-level evidence.

<<<<<<< SEARCH
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
=======
        calibration = 1.22775 * (
            0.922242487 + 0.077757513 * view_agreement
        )
>>>>>>> REPLACE