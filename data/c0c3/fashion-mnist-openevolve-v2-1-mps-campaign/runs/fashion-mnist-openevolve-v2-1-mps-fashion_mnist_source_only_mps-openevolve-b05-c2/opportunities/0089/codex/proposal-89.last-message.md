MECHANISM: Concave margin-adaptive calibration

HYPOTHESIS: Adding a −0.001 batch-mean-centered quadratic margin term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22206240844726563.

INTENDED_EDIT: Extend the best linear confidence scaling with a zero-mean concave correction that further softens low-margin predictions while tapering sharpening for extreme high margins.

EVIDENCE: Linear margin adaptation reduced cross-entropy from 0.2223743351 to 0.2220624084 without changing accuracy, while increasingly fine slope adjustments have nearly plateaued; this motivates testing curvature in the order-preserving calibration function.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.017065 * (
            margin - margin.mean()
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
=======
        centered_margin = margin - margin.mean()
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.001 * (centered_square - centered_square.mean())
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
>>>>>>> REPLACE