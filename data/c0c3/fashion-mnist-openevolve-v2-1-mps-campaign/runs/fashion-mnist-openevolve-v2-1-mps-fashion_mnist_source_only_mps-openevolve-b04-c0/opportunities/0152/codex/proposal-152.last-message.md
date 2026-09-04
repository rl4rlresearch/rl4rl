MECHANISM: Bisection refinement of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.39739990234375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300964355468 while remaining below the degradation observed at 13.39794921875%.

INTENDED_EDIT: Increase total translated-view weight from 13.3968505859375% to 13.39739990234375%, assigning 6.698699951171875% to each vertical shift.

EVIDENCE: The 13.3968505859375% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.866031494140625 * ensemble_logits
            + 0.0669842529296875 * down_logits
            + 0.0669842529296875 * up_logits
        )
=======
        translation_refined_logits = (
            0.8660260009765625 * ensemble_logits
            + 0.06698699951171875 * down_logits
            + 0.06698699951171875 * up_logits
        )
>>>>>>> REPLACE