MECHANISM: Best-verified agreement-conditioned logit calibration

HYPOTHESIS: Restoring the verified 7.788524% disagreement adjustment will preserve all 9,287 pooled predictions and reduce cross-entropy from 0.20668936004638672 to approximately 0.20668935852050782.

INTENDED_EDIT: Replace the current 7.708% agreement-conditioning strength with the best verified 7.788524% setting.

EVIDENCE: Reference Designs 1 and 3 independently achieved the highest available validation_score using this calibration; its positive per-image scaling cannot change pooled argmax predictions.

<<<<<<< SEARCH
        calibration = 1.22775 * (
            0.92292 + 0.07708 * view_agreement
        )
=======
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
>>>>>>> REPLACE