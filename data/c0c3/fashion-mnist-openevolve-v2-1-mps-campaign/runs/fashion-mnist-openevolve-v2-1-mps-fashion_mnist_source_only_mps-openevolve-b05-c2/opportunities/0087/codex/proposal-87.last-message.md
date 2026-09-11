MECHANISM: Quadratic-vertex margin calibration

HYPOTHESIS: A slope of 0.01706436 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22206240844726563.

INTENDED_EDIT: Replace the current 0.02 confidence-adaptation slope with the quadratic-interpolated local optimum.

EVIDENCE: Slopes 0.017, 0.01705, and 0.0175 yielded cross-entropies 0.2220624153137207, 0.222062410736084, and 0.22206263122558595; their quadratic vertex is approximately 0.01706436, immediately beside the best verified 0.017065 result.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.02 * (
            margin - margin.mean()
        )
=======
        confidence_scale = (10500.0 / 9564.0) + 0.01706436 * (
            margin - margin.mean()
        )
>>>>>>> REPLACE