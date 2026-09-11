MECHANISM: Locally quadratic temperature calibration

HYPOTHESIS: A 1.227325 evaluation-logit scale will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2067065514.

INTENDED_EDIT: Increase only the final center-biased pooled-logit scale from 1.225825 to the estimated local cross-entropy minimum of 1.227325.

EVIDENCE: Raising the scale from 1.225 to 1.225825 reduced cross-entropy from 0.2067071121 to 0.2067065514 without changing predictions; combining that measured slope with the curvature of the prior 1.125–1.225 calibration series estimates the minimum near 1.2273.

<<<<<<< SEARCH
        return 1.225825 * pooled_logits
=======
        return 1.227325 * pooled_logits
>>>>>>> REPLACE