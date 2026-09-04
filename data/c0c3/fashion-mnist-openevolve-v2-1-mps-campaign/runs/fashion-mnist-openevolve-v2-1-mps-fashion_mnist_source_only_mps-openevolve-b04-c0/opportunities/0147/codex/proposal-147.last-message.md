MECHANISM: Bisection of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.384765625% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436306648254394 while remaining below the degradation boundary observed at 13.40234375%.

INTENDED_EDIT: Increase total translated-view weight from 13.3671875% to 13.384765625%, assigning 6.6923828125% to each vertical shift.

EVIDENCE: The 13.3671875% blend achieved the best cross-entropy, while 13.40234375% sharply worsened it; their midpoint is the most informative probe for localizing the apparent discrete confidence-gating boundary.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.866328125 * ensemble_logits
            + 0.0668359375 * down_logits
            + 0.0668359375 * up_logits
        )
=======
        translation_refined_logits = (
            0.86615234375 * ensemble_logits
            + 0.066923828125 * down_logits
            + 0.066923828125 * up_logits
        )
>>>>>>> REPLACE