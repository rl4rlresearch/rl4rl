MECHANISM: Quadratic-vertex sextic confidence calibration

HYPOTHESIS: A centered sextic coefficient of −0.000002667 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195066108703612.

INTENDED_EDIT: Restore the best verified quintic coefficient and add a sextic term at the quadratic optimum interpolated from the verified zero, −0.00000145, and −0.00000290 results.

EVIDENCE: Equal sextic steps improved cross-entropy by 0.00000526657 and then 0.00000133438 with unchanged accuracy; the diminishing improvement implies positive local curvature and places the fitted minimum near −0.000002667.

<<<<<<< SEARCH
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001094982902 * (centered_quintic - centered_quintic.mean())
        )
=======
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        centered_sextic = centered_cube.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
            - 0.000002667 * (centered_sextic - centered_sextic.mean())
        )
>>>>>>> REPLACE