MECHANISM: Local quadratic temperature refinement

HYPOTHESIS: Setting the calibration scale to 1.24748 will preserve all 9,325 argmax predictions while reducing validation cross-entropy below 0.1929895706.

INTENDED_EDIT: Increase only the positive post-pooling calibration multiplier from 1.24739 to the local quadratic optimum of 1.24748.

EVIDENCE: Scales 1.24003, 1.24623, and 1.24739 successively reduced cross-entropy without changing correctness; the two nearest measured intervals estimate the minimum near 1.24748.

<<<<<<< SEARCH
        calibration = 1.24739 * (
=======
        calibration = 1.24748 * (
>>>>>>> REPLACE