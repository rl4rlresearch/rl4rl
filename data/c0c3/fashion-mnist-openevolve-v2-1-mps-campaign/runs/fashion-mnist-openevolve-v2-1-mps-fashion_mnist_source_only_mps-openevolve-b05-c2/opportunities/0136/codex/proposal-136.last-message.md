MECHANISM: Best-verified centered-quintic confidence calibration

HYPOTHESIS: Using −0.00001092570151 will preserve all 9,192 correct predictions and reduce validation cross-entropy below the current 0.22195726585388184.

INTENDED_EDIT: Replace the evaluation-time centered-quintic coefficient with the best verified value.

EVIDENCE: Reference Designs 2 and 3 achieved the highest available score using −0.00001092570151, with 9,192 correct predictions and cross-entropy 0.22195726203918456; every verified ultra-local probe around it was worse.

<<<<<<< SEARCH
            - 0.000010901574 * (centered_quintic - centered_quintic.mean())
=======
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE