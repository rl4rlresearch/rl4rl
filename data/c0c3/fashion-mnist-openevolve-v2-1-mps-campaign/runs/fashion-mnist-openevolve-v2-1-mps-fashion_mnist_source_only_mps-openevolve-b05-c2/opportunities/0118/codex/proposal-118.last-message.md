MECHANISM: Centered quintic margin-adaptive confidence calibration

HYPOTHESIS: Restoring the best verified quartic coefficient and adding a −0.0000082332 centered quintic term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22196439743041993.

INTENDED_EDIT: Restore the qualified −0.000082332 quartic coefficient and extend evaluation-time calibration with a quintic term one-tenth its magnitude.

EVIDENCE: The −0.000082332 quartic design achieved the best verified cross-entropy with unchanged accuracy, and the preceding cubic-to-quartic extension improved cross-entropy after same-degree coefficient refinement plateaued.

<<<<<<< SEARCH
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.00008299 * (centered_quartic - centered_quartic.mean())
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
            - 0.0000082332 * (centered_quintic - centered_quintic.mean())
        )
>>>>>>> REPLACE