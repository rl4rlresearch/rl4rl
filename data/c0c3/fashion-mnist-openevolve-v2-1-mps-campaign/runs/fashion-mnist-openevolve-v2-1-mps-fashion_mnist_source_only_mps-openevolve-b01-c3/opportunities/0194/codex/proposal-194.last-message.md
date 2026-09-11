MECHANISM: Updated local quadratic temperature refinement

HYPOTHESIS: Setting the calibration scale to 1.24754 will preserve all 9,325 argmax predictions while reducing validation cross-entropy below 0.1929895683.

INTENDED_EDIT: Increase only the positive post-pooling calibration multiplier from 1.24623 to 1.24754.

EVIDENCE: Calibration scales 1.24623, 1.24739, and 1.24748 successively reduced cross-entropy without changing correctness; incorporating the latest measurement places the fitted local minimum near 1.24754.

<<<<<<< SEARCH
        calibration = 1.24623 * (
=======
        calibration = 1.24754 * (
>>>>>>> REPLACE