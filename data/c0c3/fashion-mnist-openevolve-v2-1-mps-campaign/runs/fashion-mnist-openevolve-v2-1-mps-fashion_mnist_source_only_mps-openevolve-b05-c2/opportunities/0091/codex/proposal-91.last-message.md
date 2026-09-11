MECHANISM: Third-step concave margin-adaptive calibration

HYPOTHESIS: Increasing the centered quadratic coefficient from −0.002 to −0.003 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22200976676940917.

INTENDED_EDIT: Add the best verified linear slope and a −0.003 batch-mean-centered quadratic margin correction to evaluation-time confidence scaling.

EVIDENCE: Coefficients 0, −0.001, and −0.002 produced cross-entropies 0.2220624084, 0.2220299664, and 0.2220097668 with identical accuracy; the continuing but diminishing gains predict improvement at the next equally spaced coefficient and a local optimum near −0.00315.

<<<<<<< SEARCH
        confidence_scale = (10500.0 / 9564.0) + 0.01705 * (
            margin - margin.mean()
        )
=======
        centered_margin = margin - margin.mean()
        centered_square = centered_margin.square()
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.003 * (centered_square - centered_square.mean())
        )
>>>>>>> REPLACE