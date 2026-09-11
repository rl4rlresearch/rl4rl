MECHANISM: Interpolated quintic confidence calibration

HYPOTHESIS: Setting the centered quintic coefficient to −0.000010901574 will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22195767936706542.

INTENDED_EDIT: Replace the current quintic calibration coefficient with the best verified interpolated coefficient.

EVIDENCE: Reference Design 2 used −0.000010901574 and achieved 9,192 correct with cross-entropy 0.22195726585388184, strictly improving the current design at identical accuracy.

<<<<<<< SEARCH
            - 0.0000082332 * (centered_quintic - centered_quintic.mean())
=======
            - 0.000010901574 * (centered_quintic - centered_quintic.mean())
>>>>>>> REPLACE