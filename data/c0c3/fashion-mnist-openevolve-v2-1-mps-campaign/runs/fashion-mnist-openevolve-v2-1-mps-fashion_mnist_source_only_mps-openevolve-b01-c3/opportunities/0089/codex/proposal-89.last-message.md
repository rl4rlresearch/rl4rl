MECHANISM: Accuracy-boundary bisection for center-biased view pooling

HYPOTHESIS: A 1.546875× center-offset weight will preserve 9,287 correct predictions while reducing validation cross-entropy below 0.2067073055.

INTENDED_EDIT: Increase the unshifted offset weight from 1.53125× to the midpoint 1.546875× and update the pooling normalization.

EVIDENCE: A 1.53125× weight retained 9,287 correct, while 1.5625× lost one prediction but achieved slightly lower cross-entropy; their midpoint is the most informative refinement of the accuracy boundary.

<<<<<<< SEARCH
        pooled_logits = (
            1.53125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.53125
=======
        pooled_logits = (
            1.546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.546875
>>>>>>> REPLACE