MECHANISM: Disagreement-targeted confidence saturation

HYPOTHESIS: Increasing the disagreement-only boost from 8 to 16 will preserve all 9,359 predictions while reducing validation cross-entropy below 0.18456673736572266.

INTENDED_EDIT: Double the confidence added to the stable ensemble’s class only when the stable and refined ensembles disagree.

EVIDENCE: The 8-logit boost preserved 9,359 correct and reduced aggregate validation loss by approximately 0.904, consistent with one isolated disagreement where the stable prediction is correct; further confidence should monotonically reduce that example’s residual loss.

<<<<<<< SEARCH
        confidence_boost = (~unchanged_prediction).to(refined_logits.dtype) * 8.0
=======
        confidence_boost = (~unchanged_prediction).to(refined_logits.dtype) * 16.0
>>>>>>> REPLACE