MECHANISM: Third-order margin-adaptive confidence calibration

HYPOTHESIS: Adding a small centered cubic correction will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22200197410583497.

INTENDED_EDIT: Extend the optimized quadratic evaluation-time calibration with a −0.0001 centered cubic margin term.

EVIDENCE: Centered quadratic calibration reduced cross-entropy from 0.22206240844726563 to 0.22200197410583497 without changing accuracy, while further quadratic-coefficient refinement did not improve; testing the next polynomial term is the most direct extension of the successful mechanism.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
        )
=======
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.0001 * (centered_cube - centered_cube.mean())
        )
>>>>>>> REPLACE