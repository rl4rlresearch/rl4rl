MECHANISM: Third-step centered cubic confidence calibration

HYPOTHESIS: A −0.0003 centered cubic term will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22198370323181152.

INTENDED_EDIT: Replace the current weak quadratic-only calibration with the best-qualified quadratic coefficient and the next centered cubic coefficient.

EVIDENCE: Cubic coefficients 0, −0.0001, and −0.0002 successively reduced cross-entropy with identical accuracy; the prior −0.0003 verification timed out without producing contrary performance evidence.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.001 * (centered_square - centered_square.mean())
        )
=======
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.0003 * (centered_cube - centered_cube.mean())
        )
>>>>>>> REPLACE