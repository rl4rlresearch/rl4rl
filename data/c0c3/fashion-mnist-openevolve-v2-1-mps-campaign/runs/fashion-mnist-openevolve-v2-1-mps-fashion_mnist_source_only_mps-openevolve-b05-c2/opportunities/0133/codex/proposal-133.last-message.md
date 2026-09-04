MECHANISM: Symmetric ultra-local quintic calibration probe

HYPOTHESIS: A centered quintic coefficient of −0.00001092667725489 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726203918456.

INTENDED_EDIT: Reflect the unsuccessful −0.00001092472576511 probe across the best verified coefficient, −0.00001092570151, testing the opposite side at equal distance.

EVIDENCE: The current coefficient has the lowest verified cross-entropy; the nearest less-negative probe was worse at identical accuracy, so its untested symmetric counterpart is the most informative local refinement.

<<<<<<< SEARCH
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
=======
            - 0.00001092667725489 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE