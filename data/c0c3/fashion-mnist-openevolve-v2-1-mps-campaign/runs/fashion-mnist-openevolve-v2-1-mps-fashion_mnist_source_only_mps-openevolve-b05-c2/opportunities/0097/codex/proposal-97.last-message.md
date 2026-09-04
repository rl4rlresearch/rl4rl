MECHANISM: Updated quadratic-vertex confidence calibration

HYPOTHESIS: A centered quadratic coefficient of −0.00311336 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22200197639465333.

INTENDED_EDIT: Move the evaluation-time quadratic calibration coefficient from −0.00311057 to the local optimum fitted using the newest verification result.

EVIDENCE: Coefficients −0.003, −0.00311057, and −0.003125 yielded cross-entropies 0.22200205307006837, 0.22200197639465333, and 0.22200197715759276 with identical accuracy; quadratic interpolation places the updated minimum near −0.00311336.

<<<<<<< SEARCH
            - 0.00311057 * (centered_square - centered_square.mean())
=======
            - 0.00311336 * (centered_square - centered_square.mean())
>>>>>>> REPLACE