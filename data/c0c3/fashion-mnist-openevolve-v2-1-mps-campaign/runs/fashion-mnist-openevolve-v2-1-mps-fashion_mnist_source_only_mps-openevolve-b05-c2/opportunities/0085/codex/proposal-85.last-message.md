MECHANISM: Fine-grained margin-slope calibration

HYPOTHESIS: A slope of 0.01705 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220624153137207.

INTENDED_EDIT: Increase the batch-mean-centered confidence-adaptation slope from 0.017 to 0.01705.

EVIDENCE: Slopes 0.017, 0.0175, and 0.02 yielded cross-entropies 0.2220624153, 0.2220626312, and 0.2220722351; local quadratic interpolation places the minimum near 0.01705–0.01706.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.017 * (
=======
        confidence_scale = (10500.0 / 9564.0) + 0.01705 * (
>>>>>>> REPLACE