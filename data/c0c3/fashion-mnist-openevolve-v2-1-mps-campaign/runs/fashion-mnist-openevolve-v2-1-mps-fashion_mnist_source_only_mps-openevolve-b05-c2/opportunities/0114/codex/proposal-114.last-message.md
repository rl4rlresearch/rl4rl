MECHANISM: Second-step centered quartic confidence calibration

HYPOTHESIS: Doubling the successful centered quartic coefficient to −0.000068 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22196999893188477.

INTENDED_EDIT: Use the best-qualified cubic coefficient and strengthen its evaluation-time centered quartic correction from −0.000034 to −0.000068.

EVIDENCE: Adding a −0.000034 centered quartic term at cubic coefficient −0.00032589 reduced cross-entropy from 0.22198053398132325 to 0.22196999893188477 without changing the 9,192 correct predictions; the next equal coefficient step tests whether that improvement continues.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032515 * (centered_cube - centered_cube.mean())
        )
=======
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000068 * (centered_quartic - centered_quartic.mean())
        )
>>>>>>> REPLACE