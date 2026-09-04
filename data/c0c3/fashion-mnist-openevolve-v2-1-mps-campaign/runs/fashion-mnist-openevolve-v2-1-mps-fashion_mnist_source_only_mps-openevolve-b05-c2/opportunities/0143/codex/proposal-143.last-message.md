MECHANISM: Eighth-order margin confidence calibration

HYPOTHESIS: Adding a centered octic coefficient of −0.0000000477 to the best verified lower-order calibration will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194767456054687.

INTENDED_EDIT: Restore the best verified sextic and septic coefficients, then add a conservative centered-octic residual term.

EVIDENCE: The optimized septic term reduced cross-entropy from 0.2219506046295166 to 0.22194767456054687 without changing accuracy; the successful initial sextic and septic coefficients were each approximately 0.133 times the preceding coefficient, motivating an octic probe of the same relative scale.

<<<<<<< SEARCH
        centered_quintic = centered_margin * centered_quartic
        centered_sextic = centered_cube.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151
            * (centered_quintic - centered_quintic.mean())
            - 0.00000290 * (centered_sextic - centered_sextic.mean())
        )
=======
        centered_quintic = centered_margin * centered_quartic
        centered_sextic = centered_cube.square()
        centered_septic = centered_margin * centered_sextic
        centered_octic = centered_quartic.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
            - 0.000002667 * (centered_sextic - centered_sextic.mean())
            - 0.000000359362196 * (centered_septic - centered_septic.mean())
            - 0.0000000477 * (centered_octic - centered_octic.mean())
        )
>>>>>>> REPLACE