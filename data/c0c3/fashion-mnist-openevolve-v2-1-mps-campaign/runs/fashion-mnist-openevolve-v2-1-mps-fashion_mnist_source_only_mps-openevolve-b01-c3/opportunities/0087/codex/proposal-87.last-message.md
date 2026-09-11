MECHANISM: Boundary-refined center-biased full-view pooling

HYPOTHESIS: A 1.5625× center-offset weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.206707797.

INTENDED_EDIT: Retain all ten views, weight the unshifted offset 1.5625×, normalize the pooled logits, and use the best-verified 1.225 evaluation scale.

EVIDENCE: A 1.5× center weight achieved 9,287 correct at 0.206707797 cross-entropy, while 1.625× reduced cross-entropy slightly but lost one prediction; 1.5625× is the informative midpoint at that accuracy boundary.

<<<<<<< SEARCH
        return 1.125 * view_logits.mean(dim=(0, 1))
=======
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5625 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5625
        return 1.225 * pooled_logits
>>>>>>> REPLACE