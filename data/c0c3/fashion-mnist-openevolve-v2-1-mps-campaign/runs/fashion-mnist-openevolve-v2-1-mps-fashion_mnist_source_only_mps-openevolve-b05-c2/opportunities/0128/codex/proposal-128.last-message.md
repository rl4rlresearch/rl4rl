MECHANISM: Quadratic-vertex quintic confidence calibration

HYPOTHESIS: A centered quintic coefficient of −0.00001092871744875 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.

INTENDED_EDIT: Add a centered quintic margin term at the local quadratic optimum while retaining the verified quartic calibration.

EVIDENCE: Coefficients −0.000010901574, −0.00001092570151, and −0.00001094982902 produced cross-entropies 0.22195726585388184, 0.22195726203918456, and 0.22195726432800292 with identical accuracy; quadratic interpolation places the minimum near −0.00001092871744875.

<<<<<<< SEARCH
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
        )
=======
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092871744875
            * (centered_quintic - centered_quintic.mean())
        )
>>>>>>> REPLACE