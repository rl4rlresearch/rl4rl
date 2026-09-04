MECHANISM: Decision-invariant near-optimal temperature sharpening

HYPOTHESIS: Scaling the linear-recency ten-view logits by 1.225 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2071991.

INTENDED_EDIT: Increase only the final evaluation logit scale from 1.10 to 1.225, leaving training, parameters, and view pooling unchanged.

EVIDENCE: Successive scales of 1.10, 1.125, and 1.175 preserved 9,287 correct while reducing cross-entropy to 0.2096186, 0.2085474, and 0.2071991; the diminishing gains suggest testing 1.225 near the implied calibration optimum.

<<<<<<< SEARCH
        return 1.10 * view_logits.mean(dim=(0, 1))
=======
        return 1.225 * view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE