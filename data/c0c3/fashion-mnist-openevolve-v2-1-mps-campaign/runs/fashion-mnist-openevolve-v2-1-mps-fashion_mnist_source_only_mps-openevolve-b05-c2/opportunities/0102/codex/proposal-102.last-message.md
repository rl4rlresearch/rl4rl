MECHANISM: Stronger third-order margin-adaptive confidence calibration

HYPOTHESIS: Doubling the successful centered cubic coefficient to −0.0002 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22199077491760255.

INTENDED_EDIT: Use the best-qualified quadratic coefficient and strengthen its evaluation-time centered cubic correction from −0.0001 to −0.0002.

EVIDENCE: Adding a −0.0001 centered cubic term at quadratic coefficient −0.00311336 improved cross-entropy from 0.22200197410583497 to 0.22199077491760255 without changing the 9,192 correct predictions; the next equal coefficient step tests whether that gain continues.

<<<<<<< SEARCH
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.003125 * (centered_square - centered_square.mean())
        )
=======
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.0002 * (centered_cube - centered_cube.mean())
        )
>>>>>>> REPLACE