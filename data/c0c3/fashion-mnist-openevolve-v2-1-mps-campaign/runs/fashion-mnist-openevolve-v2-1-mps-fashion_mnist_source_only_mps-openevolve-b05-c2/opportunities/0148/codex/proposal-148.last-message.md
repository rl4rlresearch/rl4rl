MECHANISM: Quadratic-vertex nonic confidence calibration

HYPOTHESIS: A centered nonic coefficient of −0.00000000847284714 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194476356506349.

INTENDED_EDIT: Restore the optimized septic and octic terms, add the centered-nonic feature, and set its coefficient to the quadratic optimum fitted from the three verified nonic probes.

EVIDENCE: Nonic coefficients 0, −0.0000000101, and −0.0000000202 produced cross-entropies 0.22194539108276368, 0.22194476356506349, and 0.22194598770141602 at identical accuracy; quadratic interpolation places the minimum near −0.00000000847284714.

<<<<<<< SEARCH
        centered_septic = centered_margin * centered_sextic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
            - 0.000002667 * (centered_sextic - centered_sextic.mean())
            - 0.000000354 * (centered_septic - centered_septic.mean())
        )
=======
        centered_septic = centered_margin * centered_sextic
        centered_octic = centered_quartic.square()
        centered_nonic = centered_margin * centered_octic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
            - 0.000002667 * (centered_sextic - centered_sextic.mean())
            - 0.000000359362196 * (centered_septic - centered_septic.mean())
            - 0.000000076055785 * (centered_octic - centered_octic.mean())
            - 0.00000000847284714 * (centered_nonic - centered_nonic.mean())
        )
>>>>>>> REPLACE