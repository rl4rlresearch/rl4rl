MECHANISM: Locally optimized quadratic confidence calibration

HYPOTHESIS: Using −0.00311057 will preserve all 9,192 correct predictions and reduce validation cross-entropy below the current 0.22200976676940917.

INTENDED_EDIT: Replace the evaluation-time centered quadratic coefficient −0.002 with the best verified coefficient, −0.00311057.

EVIDENCE: Reference Design 2 directly verified 9,192 correct predictions and 0.22200197639465333 cross-entropy with this coefficient, outperforming the current design.

<<<<<<< SEARCH
            - 0.002 * (centered_square - centered_square.mean())
=======
            - 0.00311057 * (centered_square - centered_square.mean())
>>>>>>> REPLACE