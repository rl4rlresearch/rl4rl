MECHANISM: Best-verified agreement-conditioned logit calibration

HYPOTHESIS: Restoring the verified 7.788524% disagreement adjustment will preserve all 9,287 pooled predictions and reduce validation cross-entropy from 0.20668935928344725 to approximately 0.20668935852050782.

INTENDED_EDIT: Replace the current 7.727% agreement-conditioning strength with the best verified 7.788524% setting while leaving training and pooled predictions unchanged.

EVIDENCE: Reference Design 3 achieved the highest available validation_score with the same 9,287 correct predictions; the positive per-image calibration scale cannot change pooled argmax predictions.

<<<<<<< SEARCH
        calibration = 1.22775 * (0.92273 + 0.07727 * view_agreement)
=======
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
>>>>>>> REPLACE