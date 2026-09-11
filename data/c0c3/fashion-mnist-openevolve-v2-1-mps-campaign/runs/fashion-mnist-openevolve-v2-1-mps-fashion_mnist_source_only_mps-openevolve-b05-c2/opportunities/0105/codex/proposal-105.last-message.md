MECHANISM: Local cubic-coefficient vertex calibration

HYPOTHESIS: A centered cubic coefficient of −0.00032515 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22198066902160646.

INTENDED_EDIT: Move the evaluation-time centered cubic correction from −0.0003 to the interpolated local optimum, −0.00032515.

EVIDENCE: Cubic coefficients −0.0001, −0.0002, and −0.0003 produced cross-entropies 0.22199077491760255, 0.22198370323181152, and 0.22198066902160646 with identical accuracy; quadratic interpolation of these equally spaced results places the minimum near −0.00032515.

<<<<<<< SEARCH
            - 0.0003 * (centered_cube - centered_cube.mean())
=======
            - 0.00032515 * (centered_cube - centered_cube.mean())
>>>>>>> REPLACE