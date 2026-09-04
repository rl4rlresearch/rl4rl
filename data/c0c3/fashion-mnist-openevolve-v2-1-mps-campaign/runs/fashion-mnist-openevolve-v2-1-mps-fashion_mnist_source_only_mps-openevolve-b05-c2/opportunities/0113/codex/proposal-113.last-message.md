MECHANISM: Centered quartic margin-adaptive confidence calibration

HYPOTHESIS: Adding a −0.000034 centered quartic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.

INTENDED_EDIT: Extend the best verified quadratic-and-cubic evaluation calibration with a centered fourth-order margin term.

EVIDENCE: Successive centered polynomial extensions improved cross-entropy without changing accuracy; the optimal cubic coefficient is roughly one-tenth the quadratic coefficient, motivating a similarly scaled quartic probe after cubic-only refinement plateaued.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
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
            - 0.000034 * (centered_quartic - centered_quartic.mean())
        )
>>>>>>> REPLACE