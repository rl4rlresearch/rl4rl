MECHANISM: Centered sextic margin-adaptive confidence calibration

HYPOTHESIS: Restoring the best verified quintic coefficient and adding a −0.0000010901574 centered sextic term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22195726585388184.

INTENDED_EDIT: Replace the overshot quintic coefficient with −0.000010901574 and extend evaluation-time confidence calibration with a centered sixth-order margin term.

EVIDENCE: Reference Design 2 achieved the best verified score at −0.000010901574; the successful quartic-to-quintic extension supports probing the next polynomial degree, while the previous sextic attempt timed out without producing contrary validation evidence.

<<<<<<< SEARCH
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.0000164664 * (centered_quintic - centered_quintic.mean())
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