MECHANISM: Best-qualified quadratic confidence calibration

HYPOTHESIS: Using the qualified −0.00311336 coefficient will preserve all 9,192 correct predictions and reduce validation cross-entropy from 0.22200976676940917 to approximately 0.22200197410583497.

INTENDED_EDIT: Replace the current −0.002 centered quadratic confidence-calibration coefficient with −0.00311336.

EVIDENCE: Reference Design 2 verified 9,192 correct predictions and the best reported cross-entropy, 0.22200197410583497, using this exact coefficient.

<<<<<<< SEARCH
            - 0.002 * (centered_square - centered_square.mean())
=======
            - 0.00311336 * (centered_square - centered_square.mean())
>>>>>>> REPLACE