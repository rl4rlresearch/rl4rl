MECHANISM: Boundary-refined center-biased full-view pooling

HYPOTHESIS: A 1.53125× center-offset weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.206707797.

INTENDED_EDIT: Restore linear-recency tail averaging, retain all ten evaluation views, weight the unshifted offset 1.53125×, and apply 1.225 logit scaling.

EVIDENCE: A 1.5× center weight retained 9,287 correct, while 1.5625× lost one prediction but slightly lowered cross-entropy; their midpoint is the most informative refinement of that accuracy boundary.

<<<<<<< SEARCH
        return view_logits.mean(dim=(0, 1))
=======
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.53125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.53125
        return 1.225 * pooled_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        recency_power = 0.5
=======
        recency_power = 1.0
>>>>>>> REPLACE