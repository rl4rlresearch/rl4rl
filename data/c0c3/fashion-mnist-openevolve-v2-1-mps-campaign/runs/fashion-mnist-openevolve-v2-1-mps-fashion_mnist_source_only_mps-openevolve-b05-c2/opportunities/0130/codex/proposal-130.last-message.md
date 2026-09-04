MECHANISM: Symmetric micro-bracket quadratic interpolation

HYPOTHESIS: A centered quintic coefficient of −0.00001092472576511 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.

INTENDED_EDIT: Replace the current overshot quintic coefficient with the quadratic vertex fitted from the closest symmetric probes around the best verified coefficient.

EVIDENCE: The best coefficient, −0.00001092570151, achieved cross-entropy 0.22195726203918456; equal-distance probes at −0.00001092268557125 and −0.00001092871744875 were both worse, and their asymmetric losses place the interpolated minimum slightly toward the less-negative probe.

<<<<<<< SEARCH
            - 0.00001094982902 * (centered_quintic - centered_quintic.mean())
=======
            - 0.00001092472576511 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE