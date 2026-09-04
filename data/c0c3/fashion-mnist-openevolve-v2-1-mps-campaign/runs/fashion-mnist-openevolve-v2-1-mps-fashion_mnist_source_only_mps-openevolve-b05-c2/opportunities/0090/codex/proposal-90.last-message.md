MECHANISM: Stronger concave margin-adaptive calibration

HYPOTHESIS: Doubling the verified quadratic coefficient to −0.002 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220299663543701.

INTENDED_EDIT: Extend the best linear calibration with a stronger batch-mean-centered concave quadratic margin correction.

EVIDENCE: Adding a −0.001 centered quadratic term improved cross-entropy from 0.22206240844726563 to 0.2220299663543701 without changing accuracy; doubling that first successful curvature step directly tests whether the gain continues.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.01706436 * (
            margin - margin.mean()
        )
=======
        centered_margin = margin - margin.mean()
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.002 * (centered_square - centered_square.mean())
        )
>>>>>>> REPLACE