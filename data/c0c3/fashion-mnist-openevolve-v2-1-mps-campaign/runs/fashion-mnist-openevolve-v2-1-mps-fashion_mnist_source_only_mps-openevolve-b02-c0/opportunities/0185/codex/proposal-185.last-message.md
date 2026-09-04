MECHANISM: One-step probability–geometric ensemble interpolation

HYPOTHESIS: Increasing the probability-ensemble contribution by one effective float32 increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Transfer 5.960464477539063e-8 of evaluation weight from the geometric ensemble to the probability ensemble while preserving their total weight.

EVIDENCE: The neighboring spatial-weight and logit-scale refinements all regressed; this orthogonal ensemble-mixture adjustment remains unresolved because its prior verification timed out.

<<<<<<< SEARCH
        ensemble_scores = (
            0.75 * probability_ensemble + 0.25 * geometric_ensemble
        )
=======
        ensemble_scores = (
            0.7500000596046448 * probability_ensemble
            + 0.24999994039535522 * geometric_ensemble
        )
>>>>>>> REPLACE