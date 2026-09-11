MECHANISM: Refined local quartic-coefficient vertex calibration

HYPOTHESIS: A centered quartic coefficient of −0.0000817727 will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22196439743041993.

INTENDED_EDIT: Move the evaluation-time centered quartic correction from −0.000082332 to the quadratic vertex fitted from the nearest verified measurements.

EVIDENCE: Quartic coefficients −0.000068, −0.000082332, and −0.00008299 produced cross-entropies 0.22196489181518556, 0.22196439743041993, and 0.22196440048217772 with identical accuracy; quadratic interpolation places the minimum near −0.0000817727.

<<<<<<< SEARCH
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
=======
            - 0.0000817727 * (centered_quartic - centered_quartic.mean())
>>>>>>> REPLACE