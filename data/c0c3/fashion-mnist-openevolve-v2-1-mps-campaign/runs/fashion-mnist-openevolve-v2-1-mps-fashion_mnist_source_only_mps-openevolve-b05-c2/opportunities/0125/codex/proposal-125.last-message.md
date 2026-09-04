MECHANISM: Refined local quintic-coefficient vertex calibration

HYPOTHESIS: Moving the centered quintic coefficient to −0.00001087744649 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726585388184.

INTENDED_EDIT: Replace the current evaluation-time quintic coefficient with the quadratic vertex fitted from the three nearest verified quintic measurements.

EVIDENCE: Quintic coefficients −0.0000082332, −0.000010901574, and −0.0000164664 yielded cross-entropies 0.22195767936706542, 0.22195726585388184, and 0.22195911331176757 with identical accuracy; local quadratic interpolation places the minimum near −0.00001087744649.

<<<<<<< SEARCH
            - 0.000010901574 * (centered_quintic - centered_quintic.mean())
=======
            - 0.00001087744649 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE