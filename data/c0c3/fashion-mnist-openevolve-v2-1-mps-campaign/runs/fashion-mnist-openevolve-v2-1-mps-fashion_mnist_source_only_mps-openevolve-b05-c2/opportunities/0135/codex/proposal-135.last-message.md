MECHANISM: Best-verified centered-quintic confidence calibration

HYPOTHESIS: Restoring −0.00001092570151 will preserve all 9,192 argmax predictions and reduce cross-entropy from 0.22195726432800292 to the verified 0.22195726203918456.

INTENDED_EDIT: Replace the overshot evaluation-time quintic coefficient with the best verified value.

EVIDENCE: The same coefficient achieved the highest available score in Reference Designs 1 and 3; every ultra-local probe around it was worse at identical accuracy.

<<<<<<< SEARCH
            - 0.00001094982902 * (centered_quintic - centered_quintic.mean())
=======
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE