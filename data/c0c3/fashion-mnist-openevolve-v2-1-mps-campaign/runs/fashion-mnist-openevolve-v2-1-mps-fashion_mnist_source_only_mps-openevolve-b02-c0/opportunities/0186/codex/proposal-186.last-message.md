MECHANISM: One-step geometric–probability ensemble interpolation

HYPOTHESIS: Decreasing the probability-ensemble contribution by one effective float32 increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Transfer 5.960464477539063e-8 of evaluation weight from the probability ensemble to the geometric ensemble while preserving their total weight.

EVIDENCE: Increasing the probability contribution by the same increment preserved correctness but worsened cross-entropy to 0.2122733238220215, making the exact opposite direction the most informative neighboring setting.

<<<<<<< SEARCH
        ensemble_scores = (
            0.75 * probability_ensemble + 0.25 * geometric_ensemble
        )
=======
        ensemble_scores = (
            0.7499999403953552 * probability_ensemble
            + 0.2500000596046448 * geometric_ensemble
        )
>>>>>>> REPLACE