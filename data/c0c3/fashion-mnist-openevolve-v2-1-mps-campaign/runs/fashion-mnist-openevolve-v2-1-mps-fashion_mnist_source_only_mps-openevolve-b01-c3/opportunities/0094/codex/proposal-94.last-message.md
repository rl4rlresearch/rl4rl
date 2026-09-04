MECHANISM: Local temperature-calibration refinement

HYPOTHESIS: A 1.22775 evaluation-logit scale will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2067060364.

INTENDED_EDIT: Increase only the final center-biased pooled-logit scale from 1.225 to 1.22775.

EVIDENCE: With identical 1.546875× center pooling, increasing the scale from 1.225 to 1.225825 and then 1.227325 preserved 9,287 correct while reducing cross-entropy from 0.2067071121 to 0.2067065514 and 0.2067060364; the measured local curvature places the estimated minimum near 1.22775.

<<<<<<< SEARCH
        return 1.225 * pooled_logits
=======
        return 1.22775 * pooled_logits
>>>>>>> REPLACE