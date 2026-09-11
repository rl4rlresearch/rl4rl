MECHANISM: Evidence-backed posterior temperature calibration

HYPOTHESIS: Changing the paired-view probability ensemble temperature from 0.78 to 0.75 will preserve all 9,315 correct predictions while lowering validation cross-entropy to approximately 0.19136.

INTENDED_EDIT: Sharpen only the final ensemble logits by changing the inference temperature to 0.75.

EVIDENCE: Reference Design 2 used the otherwise identical implementation at temperature 0.75 and achieved 9,315 correct with 0.1913601 cross-entropy, outperforming the current temperature-0.78 result of 0.1915621.

<<<<<<< SEARCH
        ).log() / 0.78
=======
        ).log() / 0.75
>>>>>>> REPLACE