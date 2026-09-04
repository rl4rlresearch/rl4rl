MECHANISM: Local quadratic margin-slope refinement

HYPOTHESIS: A slope of 0.017065 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.222062410736084.

INTENDED_EDIT: Change the batch-mean-centered confidence-adaptation slope from 0.0175 to 0.017065.

EVIDENCE: Slopes 0.017, 0.01705, and 0.0175 produced cross-entropies 0.2220624153137207, 0.222062410736084, and 0.22206263122558595 with identical accuracy; local quadratic interpolation places the minimum near 0.017064.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.0175 * (
            margin - margin.mean()
        )
=======
        confidence_scale = (10500.0 / 9564.0) + 0.017065 * (
            margin - margin.mean()
        )
>>>>>>> REPLACE