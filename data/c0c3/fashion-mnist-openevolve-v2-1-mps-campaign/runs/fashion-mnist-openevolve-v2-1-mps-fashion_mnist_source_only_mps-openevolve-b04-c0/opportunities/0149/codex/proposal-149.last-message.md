MECHANISM: Bisection refinement of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.39794921875% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436301498413085 while remaining below the degradation observed at 13.40234375%.

INTENDED_EDIT: Increase total translated-view weight from 13.3935546875% to 13.39794921875%, assigning 6.698974609375% to each vertical shift.

EVIDENCE: The 13.3935546875% blend achieved the best cross-entropy, while 13.40234375% sharply worsened it; their midpoint is the most informative probe of the apparent discrete confidence-gating boundary.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.866064453125 * ensemble_logits
            + 0.0669677734375 * down_logits
            + 0.0669677734375 * up_logits
        )
=======
        translation_refined_logits = (
            0.8660205078125 * ensemble_logits
            + 0.06698974609375 * down_logits
            + 0.06698974609375 * up_logits
        )
>>>>>>> REPLACE