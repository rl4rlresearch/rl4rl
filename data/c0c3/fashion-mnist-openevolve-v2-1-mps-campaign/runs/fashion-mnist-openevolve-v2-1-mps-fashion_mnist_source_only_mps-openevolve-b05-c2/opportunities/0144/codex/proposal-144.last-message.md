MECHANISM: Directional octic calibration continuation

HYPOTHESIS: Doubling the centered octic coefficient to −0.0000000954 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194570770263672.

INTENDED_EDIT: Restore the best verified sextic and septic coefficients, add the centered-octic feature, and continue one equal-sized octic step beyond the successful −0.0000000477 probe.

EVIDENCE: Adding −0.0000000477 octic calibration reduced cross-entropy from 0.22194767456054687 to 0.22194570770263672 without changing accuracy; an equal-step continuation is the most informative directional probe.

<<<<<<< SEARCH
        centered_quintic = centered_margin * centered_quartic
        centered_sextic = centered_cube.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
            - 0.00000145 * (centered_sextic - centered_sextic.mean())
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
            - 0.0000000954 * (centered_octic - centered_octic.mean())
        )
>>>>>>> REPLACE