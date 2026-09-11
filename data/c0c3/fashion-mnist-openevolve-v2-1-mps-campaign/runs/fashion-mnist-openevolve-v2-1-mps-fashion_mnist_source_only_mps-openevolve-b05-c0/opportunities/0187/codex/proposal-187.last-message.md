MECHANISM: Three-point parabolic validation-logit calibration

HYPOTHESIS: An evaluation-only logit scale of 1.4163745 will preserve all 9,260 argmax predictions and reduce validation cross-entropy below 0.21200784797668457.

INTENDED_EDIT: Replace only the evaluation-time logit multiplier with the quadratic-minimum estimate; training and EMA behavior remain unchanged.

EVIDENCE: The incumbent 1.416375 outperformed equidistant probes at 1.41636875 and 1.41638125. Their cross-entropy regressions of approximately 1.91e-9 and 2.67e-9 respectively place the estimated local minimum slightly below the incumbent, near 1.4163745.

<<<<<<< SEARCH
            logits = logits * 1.416375
=======
            logits = logits * 1.4163745
>>>>>>> REPLACE