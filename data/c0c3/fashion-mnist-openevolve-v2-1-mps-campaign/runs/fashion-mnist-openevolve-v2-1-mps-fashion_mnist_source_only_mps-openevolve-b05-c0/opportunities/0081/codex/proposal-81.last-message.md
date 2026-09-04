MECHANISM: Incremental inference-logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.

INTENDED_EDIT: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.26.

EVIDENCE: Increasing the multiplier from 1.20 to 1.22 and then 1.24 preserved all 9,258 correct predictions while cross-entropy fell from 0.2193095 to 0.2180025 and then 0.2168836, supporting another small upward calibration step.

<<<<<<< SEARCH
            logits = logits * 1.24
=======
            logits = logits * 1.26
>>>>>>> REPLACE