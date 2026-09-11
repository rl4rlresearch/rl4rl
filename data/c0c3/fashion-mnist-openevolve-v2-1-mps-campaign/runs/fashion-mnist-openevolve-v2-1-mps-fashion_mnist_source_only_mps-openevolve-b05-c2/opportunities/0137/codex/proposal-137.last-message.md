MECHANISM: Sixth-order margin confidence calibration

HYPOTHESIS: Adding a centered sextic coefficient of −0.00000145 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726203918456.

INTENDED_EDIT: Retain the best verified quintic coefficient and add a small centered sextic term to model residual confidence-calibration error.

EVIDENCE: Every ultra-local quintic probe around −0.00001092570151 was worse at identical accuracy, indicating that coordinate is exhausted; the existing progressively smaller higher-order coefficients motivate testing the next orthogonal centered-margin term.

<<<<<<< SEARCH
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
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
            - 0.00000145 * (centered_sextic - centered_sextic.mean())
        )
>>>>>>> REPLACE