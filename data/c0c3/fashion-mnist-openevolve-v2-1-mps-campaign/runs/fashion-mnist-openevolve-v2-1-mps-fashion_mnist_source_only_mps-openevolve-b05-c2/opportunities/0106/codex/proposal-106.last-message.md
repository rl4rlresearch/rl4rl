MECHANISM: Updated local cubic-coefficient vertex calibration

HYPOTHESIS: A centered cubic coefficient of −0.00032589 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22198053512573243.

INTENDED_EDIT: Move the evaluation-time centered cubic correction from −0.0002 to the updated interpolated optimum, −0.00032589.

EVIDENCE: Cubic coefficients −0.0002, −0.0003, and −0.00032515 produced cross-entropies 0.22198370323181152, 0.22198066902160646, and 0.22198053512573243 with identical accuracy; quadratic interpolation of these closest measurements places the minimum near −0.00032589.

<<<<<<< SEARCH
            - 0.0002 * (centered_cube - centered_cube.mean())
=======
            - 0.00032589 * (centered_cube - centered_cube.mean())
>>>>>>> REPLACE