MECHANISM: Bisection refinement of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.3975372314453125% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300659179689 while remaining below the degradation observed at 13.397674560546875%.

INTENDED_EDIT: Increase total translated-view weight from 13.39739990234375% to 13.3975372314453125%, assigning 6.69876861572265625% to each vertical shift.

EVIDENCE: The 13.39739990234375% blend achieved the best cross-entropy, while 13.397674560546875% sharply worsened it; their midpoint is the most informative probe for localizing the apparent discrete confidence-gating boundary.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8660260009765625 * ensemble_logits
            + 0.06698699951171875 * down_logits
            + 0.06698699951171875 * up_logits
        )
=======
        translation_refined_logits = (
            0.866024627685546875 * ensemble_logits
            + 0.0669876861572265625 * down_logits
            + 0.0669876861572265625 * up_logits
        )
>>>>>>> REPLACE