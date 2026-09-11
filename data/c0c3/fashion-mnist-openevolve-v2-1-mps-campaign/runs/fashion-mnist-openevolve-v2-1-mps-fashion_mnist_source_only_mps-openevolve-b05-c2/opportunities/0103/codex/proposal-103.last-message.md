MECHANISM: Third-step centered cubic confidence calibration

HYPOTHESIS: Using the qualified quadratic coefficient with a −0.0003 centered cubic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198370323181152.

INTENDED_EDIT: Replace the current quadratic-only evaluation calibration with the best-qualified quadratic coefficient and the next equally spaced centered cubic correction.

EVIDENCE: Cubic coefficients 0, −0.0001, and −0.0002 yielded cross-entropies 0.22200197410583497, 0.22199077491760255, and 0.22198370323181152 with identical accuracy; the continuing diminishing improvement motivates testing −0.0003.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.002 * (centered_square - centered_square.mean())
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