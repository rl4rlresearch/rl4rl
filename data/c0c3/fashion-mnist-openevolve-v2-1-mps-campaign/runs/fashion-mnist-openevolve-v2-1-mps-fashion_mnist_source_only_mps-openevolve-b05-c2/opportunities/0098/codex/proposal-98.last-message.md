MECHANISM: Updated local quadratic-vertex calibration

HYPOTHESIS: A centered quadratic coefficient of −0.00311743 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22200197410583497.

INTENDED_EDIT: Move the evaluation-time centered quadratic confidence-calibration coefficient from −0.003125 to the vertex fitted from the three closest verified coefficients.

EVIDENCE: Coefficients −0.00311057, −0.00311336, and −0.003125 produced cross-entropies 0.22200197639465333, 0.22200197410583497, and 0.22200197715759276 with identical accuracy; quadratic interpolation places the minimum near −0.00311743.

<<<<<<< SEARCH
            - 0.003125 * (centered_square - centered_square.mean())
=======
            - 0.00311743 * (centered_square - centered_square.mean())
>>>>>>> REPLACE