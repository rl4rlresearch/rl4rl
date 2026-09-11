MECHANISM: Interpolated evaluation-temperature calibration

HYPOTHESIS: Temperature 0.805 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472, as the improvements from 0.90 to 0.85 to 0.80 place the estimated calibration optimum near 0.805.

INTENDED_EDIT: Slightly relax evaluation-only sharpening from temperature 0.80 to 0.805; training remains unchanged.

EVIDENCE: Temperatures 0.90, 0.85, and 0.80 preserved exactly 9,328 correct while cross-entropy decreased from 0.194148 to 0.192145 to 0.191472; quadratic interpolation in inverse temperature estimates a minimum just above 0.80.

<<<<<<< SEARCH
        return fused_log_probabilities / 0.80
=======
        return fused_log_probabilities / 0.805
>>>>>>> REPLACE