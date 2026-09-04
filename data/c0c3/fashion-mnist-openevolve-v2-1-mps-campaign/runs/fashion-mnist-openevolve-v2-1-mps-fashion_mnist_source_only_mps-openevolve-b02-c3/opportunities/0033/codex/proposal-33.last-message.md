MECHANISM: Quadratic-guided posterior temperature calibration

HYPOTHESIS: Temperature 0.75 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1913707.

INTENDED_EDIT: Sharpen only the final probability-ensemble logits from temperature 0.76 to 0.75.

EVIDENCE: Temperatures 0.80, 0.78, and 0.76 preserved identical predictions while cross-entropy improved to 0.1919746, 0.1915621, and 0.1913707; those diminishing gains place the estimated minimum near 0.753.

<<<<<<< SEARCH
        ).log() / 0.76
=======
        ).log() / 0.75
>>>>>>> REPLACE