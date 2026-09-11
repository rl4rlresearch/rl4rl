MECHANISM: Updated local cubic-coefficient vertex calibration

HYPOTHESIS: A centered cubic coefficient of −0.00033082 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.

INTENDED_EDIT: Replace the current quadratic-only calibration with the qualified quadratic coefficient and the newly interpolated cubic optimum.

EVIDENCE: Cubic coefficients −0.0003, −0.00032515, and −0.00032589 produced cross-entropies 0.22198066902160646, 0.22198053512573243, and 0.22198053398132325 with identical accuracy; quadratic interpolation of these newest local measurements places the minimum near −0.00033082.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.002 * (centered_square - centered_square.mean())
        )
=======
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00033082 * (centered_cube - centered_cube.mean())
        )
>>>>>>> REPLACE