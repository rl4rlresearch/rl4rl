MECHANISM: Verified local temperature-calibration refinement

HYPOTHESIS: Scaling the center-biased pooled logits by 1.22775 will preserve all 9,287 correct predictions while reducing validation cross-entropy below 0.2067060364.

INTENDED_EDIT: Increase only the evaluation-logit scale from 1.227325 to the best-verified 1.22775.

EVIDENCE: Reference Design 2 used identical 1.546875× center-biased pooling with a 1.22775 scale and achieved 9,287 correct at 0.2067060093 cross-entropy, strictly improving the current design.

<<<<<<< SEARCH
        return 1.227325 * pooled_logits
=======
        return 1.22775 * pooled_logits
>>>>>>> REPLACE