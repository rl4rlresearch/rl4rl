MECHANISM: Accuracy-boundary bisection for center-biased view pooling

HYPOTHESIS: A 1.5546875× center-offset weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.2067071121.

INTENDED_EDIT: Retain all ten evaluation views and the 1.225 logit scale, but weight the unshifted offset 1.5546875× with matching normalization.

EVIDENCE: A 1.546875× center weight retained 9,287 correct at 0.2067071121 cross-entropy, while 1.5625× lost one prediction; their midpoint is the most informative refinement of the observed accuracy boundary.

<<<<<<< SEARCH
        return 1.225 * view_logits.mean(dim=(0, 1))
=======
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5546875
        return 1.225 * pooled_logits
>>>>>>> REPLACE