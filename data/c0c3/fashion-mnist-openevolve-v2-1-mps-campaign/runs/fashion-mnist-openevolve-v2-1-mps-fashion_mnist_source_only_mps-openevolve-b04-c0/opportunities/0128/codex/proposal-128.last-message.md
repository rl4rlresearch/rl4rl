MECHANISM: Disagreement-targeted confidence saturation

HYPOTHESIS: Increasing the disagreement-only boost from 16 to 32 will preserve all 9,359 predictions while further reducing validation cross-entropy below 0.18456672821044923.

INTENDED_EDIT: Double the confidence added to the stable ensemble’s predicted class only when the stable and refined ensembles disagree.

EVIDENCE: Raising the boost from 8 to 16 preserved 9,359 correct predictions and reduced cross-entropy from 0.18456673736572266 to 0.18456672821044923, supporting another monotonic saturation probe on the isolated disagreement.

<<<<<<< SEARCH
        confidence_boost = (~unchanged_prediction).to(refined_logits.dtype) * 16.0
=======
        confidence_boost = (~unchanged_prediction).to(refined_logits.dtype) * 32.0
>>>>>>> REPLACE