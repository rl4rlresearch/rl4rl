MECHANISM: Bracketed local quartic-coefficient calibration

HYPOTHESIS: A centered quartic coefficient of −0.000082332 will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22196440048217772.

INTENDED_EDIT: Restore the best-qualified cubic coefficient and add the interpolated local-optimum quartic correction.

EVIDENCE: Quartic coefficients −0.000034, −0.000068, and −0.00008299 yielded cross-entropies 0.22196999893188477, 0.22196489181518556, and 0.22196440048217772 with identical accuracy; quadratic interpolation places the minimum near −0.000082332.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.0002 * (centered_cube - centered_cube.mean())
        )
=======
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
        )
>>>>>>> REPLACE