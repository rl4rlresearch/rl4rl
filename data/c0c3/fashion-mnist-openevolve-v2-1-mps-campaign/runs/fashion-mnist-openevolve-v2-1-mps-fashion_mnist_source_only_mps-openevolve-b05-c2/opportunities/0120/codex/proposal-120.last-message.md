MECHANISM: Interpolated quintic confidence calibration

HYPOTHESIS: A centered quintic coefficient of −0.000010901574 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195767936706542.

INTENDED_EDIT: Restore the best verified quartic coefficient and add the quintic coefficient at the quadratic vertex fitted from the three verified quintic measurements.

EVIDENCE: Quintic coefficients 0, −0.0000082332, and −0.0000164664 produced cross-entropies 0.22196439743041993, 0.22195767936706542, and 0.22195911331176757 with identical accuracy; quadratic interpolation places the minimum near −0.000010901574.

<<<<<<< SEARCH
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000034 * (centered_quartic - centered_quartic.mean())
        )
=======
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.000010901574 * (centered_quintic - centered_quintic.mean())
        )
>>>>>>> REPLACE