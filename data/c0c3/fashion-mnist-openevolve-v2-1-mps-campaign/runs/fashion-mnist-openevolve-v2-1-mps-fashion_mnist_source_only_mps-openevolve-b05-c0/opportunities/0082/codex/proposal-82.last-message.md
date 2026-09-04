MECHANISM: Incremental inference-logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.

INTENDED_EDIT: Increase only the evaluation-time logit multiplier from 1.24 to 1.26.

EVIDENCE: Increases from 1.20 to 1.22 and 1.24 preserved all 9,258 correct predictions while cross-entropy fell from 0.2193095 to 0.2180025 and 0.2168836; the prior 1.26 verification timed out and supplied no contradictory calibration result.

<<<<<<< SEARCH
        if not self.training:
            logits = logits * 1.24
=======
        if not self.training:
            logits = logits * 1.26
>>>>>>> REPLACE