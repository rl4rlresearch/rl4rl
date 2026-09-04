MECHANISM: Best-verified centered-quintic confidence calibration

HYPOTHESIS: Replacing the current quintic coefficient with −0.00001092570151 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726585388184.

INTENDED_EDIT: Set the evaluation-time centered-quintic calibration coefficient to the best verified value.

EVIDENCE: Reference Design 2 used −0.00001092570151 and achieved 9,192 correct with cross-entropy 0.22195726203918456, strictly outperforming the current coefficient at identical accuracy.

<<<<<<< SEARCH
            - 0.000010901574 * (centered_quintic - centered_quintic.mean())
=======
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE