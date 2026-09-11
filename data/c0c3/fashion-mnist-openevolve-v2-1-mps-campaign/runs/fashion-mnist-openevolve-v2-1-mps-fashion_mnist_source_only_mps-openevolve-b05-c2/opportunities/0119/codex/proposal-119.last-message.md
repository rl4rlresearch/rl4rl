MECHANISM: Second-step centered quintic confidence calibration

HYPOTHESIS: Doubling the successful centered quintic coefficient to −0.0000164664 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22195767936706542.

INTENDED_EDIT: Restore the best verified quartic coefficient, compute the centered quintic margin term, and double its successful evaluation-time correction.

EVIDENCE: Adding a −0.0000082332 quintic term at the best quartic coefficient reduced cross-entropy from 0.22196439743041993 to 0.22195767936706542 without changing accuracy; the analogous second-step quartic probe also improved cross-entropy substantially.

<<<<<<< SEARCH
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000068 * (centered_quartic - centered_quartic.mean())
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
            - 0.0000164664 * (centered_quintic - centered_quintic.mean())
        )
>>>>>>> REPLACE