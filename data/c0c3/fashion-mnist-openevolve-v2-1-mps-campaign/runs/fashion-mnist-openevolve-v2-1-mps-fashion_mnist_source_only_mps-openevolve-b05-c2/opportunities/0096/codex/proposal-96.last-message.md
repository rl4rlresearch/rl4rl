MECHANISM: Best-verified quadratic confidence calibration

HYPOTHESIS: Setting the centered quadratic coefficient to −0.00311057 will preserve all 9,192 correct predictions and reduce validation cross-entropy below the current 0.2220299663543701.

INTENDED_EDIT: Replace the current −0.001 evaluation-time quadratic calibration coefficient with the best verified coefficient, −0.00311057.

EVIDENCE: Reference Design 2 verified 9,192 correct predictions and 0.22200197639465333 cross-entropy at −0.00311057, outperforming the current design while leaving argmax predictions unchanged.

<<<<<<< SEARCH
            - 0.001 * (centered_square - centered_square.mean())
=======
            - 0.00311057 * (centered_square - centered_square.mean())
>>>>>>> REPLACE