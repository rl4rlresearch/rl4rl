MECHANISM: Interpolated quartic confidence calibration

HYPOTHESIS: A centered quartic coefficient of −0.00008299 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22196489181518556.

INTENDED_EDIT: Restore the best-qualified cubic coefficient and add the quartic coefficient estimated from the three verified quartic measurements.

EVIDENCE: Quartic coefficients 0, −0.000034, and −0.000068 produced cross-entropies 0.22198053398132325, 0.22196999893188477, and 0.22196489181518556 with identical accuracy; quadratic interpolation places the local optimum near −0.00008299.

<<<<<<< SEARCH
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00033082 * (centered_cube - centered_cube.mean())
        )
=======
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.00008299 * (centered_quartic - centered_quartic.mean())
        )
>>>>>>> REPLACE