MECHANISM: Accuracy-preserving ensemble-weight refinement

HYPOTHESIS: A 53.807299137115478515625% original weighting will preserve 9,359 correct predictions and reduce cross-entropy below 0.18465710182189943.

INTENDED_EDIT: Move halfway from the best current weighting toward the nearest higher weighting that preserved accuracy but worsened cross-entropy.

EVIDENCE: The current 53.8072967529296875% weighting is best at 0.18465710182189943 cross-entropy, while 53.80730152130126953125% preserved 9,359 correct but increased cross-entropy; their midpoint is the most informative local calibration probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.538072967529296875 * logits + 0.461927032470703125 * flipped_logits
        )
=======
        return 1.226016 * (
            0.53807299137115478515625 * logits
            + 0.46192700862884521484375 * flipped_logits
        )
>>>>>>> REPLACE