MECHANISM: Bisection refinement of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.3968505859375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436301002502442 while remaining below the degradation observed at 13.39794921875%.

INTENDED_EDIT: Increase total translated-view weight from 13.395751953125% to 13.3968505859375%, assigning 6.69842529296875% to each vertical shift.

EVIDENCE: The 13.395751953125% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86604248046875 * ensemble_logits
            + 0.066978759765625 * down_logits
            + 0.066978759765625 * up_logits
        )
=======
        translation_refined_logits = (
            0.866031494140625 * ensemble_logits
            + 0.0669842529296875 * down_logits
            + 0.0669842529296875 * up_logits
        )
>>>>>>> REPLACE