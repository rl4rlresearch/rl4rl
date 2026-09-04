MECHANISM: Matched-order orientation power fusion

HYPOTHESIS: Applying the verified 1.174-order power mean within native/reflected pairs will exceed 9,348 correct predictions by emphasizing reliable orientation-specific evidence rather than suppressing it.

INTENDED_EDIT: Replace arithmetic orientation averaging with a 1.174-order generalized mean while retaining the verified crop fusion, weights, training, and reciprocal calibration.

EVIDENCE: Geometric orientation consensus reduced correctness to 9,345 versus 9,348 for arithmetic consensus, while super-arithmetic 1.174-order crop fusion produced the best verified design; this motivates testing the same evidence-emphasizing direction across orientations.

<<<<<<< SEARCH
            orientation_consensus = 0.5 * (
                native_probabilities + reflected_probabilities
            )
=======
            orientation_consensus = (
                0.5
                * (
                    native_probabilities.pow(fusion_power)
                    + reflected_probabilities.pow(fusion_power)
                )
            ).pow(1.0 / fusion_power)
>>>>>>> REPLACE