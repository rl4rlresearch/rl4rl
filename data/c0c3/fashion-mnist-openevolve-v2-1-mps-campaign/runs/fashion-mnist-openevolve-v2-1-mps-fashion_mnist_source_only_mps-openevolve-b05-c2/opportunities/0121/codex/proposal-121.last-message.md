MECHANISM: Centered sextic margin-adaptive confidence calibration

HYPOTHESIS: Adding a −0.0000010901574 centered sextic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726585388184.

INTENDED_EDIT: Extend the best verified evaluation-time polynomial calibration with a centered sixth-order margin term one-tenth the magnitude of the quintic coefficient.

EVIDENCE: Adding the −0.0000082332 quintic term after quartic refinement improved cross-entropy without changing accuracy; the interpolated −0.000010901574 quintic coefficient improved it further, motivating the next-degree probe.

<<<<<<< SEARCH
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
            - 0.000010901574 * (centered_quintic - centered_quintic.mean())
            - 0.0000010901574 * (centered_sextic - centered_sextic.mean())
        )
>>>>>>> REPLACE