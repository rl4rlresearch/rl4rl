MECHANISM: Refined local cubic-coefficient vertex calibration

HYPOTHESIS: A centered cubic coefficient of −0.00032687 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.

INTENDED_EDIT: Move the evaluation-time centered cubic correction from −0.00032589 to the updated local optimum, −0.00032687.

EVIDENCE: Coefficients −0.00032515, −0.00032589, and −0.00033082 produced cross-entropies 0.22198053512573243, 0.22198053398132325, and 0.22198054237365722 with identical accuracy; quadratic interpolation of these nearest measurements places the minimum near −0.00032687.

<<<<<<< SEARCH
            - 0.00032589 * (centered_cube - centered_cube.mean())
=======
            - 0.00032687 * (centered_cube - centered_cube.mean())
>>>>>>> REPLACE