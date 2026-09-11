MECHANISM: Directional sextic calibration continuation

HYPOTHESIS: Doubling the centered sextic coefficient to −0.00000290 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195199546813965.

INTENDED_EDIT: Add the centered sextic margin feature and continue one equal-sized step beyond the verified −0.00000145 coefficient.

EVIDENCE: Moving the sextic coefficient from zero to −0.00000145 reduced cross-entropy from 0.22195726203918456 to 0.22195199546813965 with unchanged accuracy, so an equal-step continuation is the most informative directional probe.

<<<<<<< SEARCH
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151
            * (centered_quintic - centered_quintic.mean())
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
            - 0.00001092570151
            * (centered_quintic - centered_quintic.mean())
            - 0.00000290 * (centered_sextic - centered_sextic.mean())
        )
>>>>>>> REPLACE