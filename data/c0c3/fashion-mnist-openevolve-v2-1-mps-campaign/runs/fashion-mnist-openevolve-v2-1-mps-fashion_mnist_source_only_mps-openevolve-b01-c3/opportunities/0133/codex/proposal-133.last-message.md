MECHANISM: Local quadratic refinement of agreement-conditioned calibration

HYPOTHESIS: A 7.727% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.

INTENDED_EDIT: Move the agreement-conditioning strength from 7.708% to the quadratic minimum estimated from the verified 0%, 7.708%, and 15% results.

EVIDENCE: Those calibration strengths produced cross-entropies of 0.20670592765808105, 0.20668936004638672, and 0.20670405883789061; local quadratic interpolation places the minimum near 7.727%, and positive scaling leaves every pooled argmax unchanged.

<<<<<<< SEARCH
        calibration = 1.22775 * (
            0.92292 + 0.07708 * view_agreement
        )
=======
        calibration = 1.22775 * (
            0.92273 + 0.07727 * view_agreement
        )
>>>>>>> REPLACE