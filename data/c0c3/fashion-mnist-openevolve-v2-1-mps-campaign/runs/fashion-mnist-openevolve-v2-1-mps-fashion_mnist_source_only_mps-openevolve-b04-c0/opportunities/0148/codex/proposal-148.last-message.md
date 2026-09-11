MECHANISM: Bisection refinement of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.3935546875% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436303024291992 while remaining below the degradation boundary observed at 13.40234375%.

INTENDED_EDIT: Increase total translated-view weight from 13.384765625% to 13.3935546875%, assigning 6.69677734375% to each vertical shift.

EVIDENCE: The 13.384765625% blend improved cross-entropy, whereas 13.40234375% sharply worsened it; their midpoint is the most informative next probe of the apparent discrete gating boundary.

<<<<<<< SEARCH
            0.86615234375 * ensemble_logits
            + 0.066923828125 * down_logits
            + 0.066923828125 * up_logits
=======
            0.866064453125 * ensemble_logits
            + 0.0669677734375 * down_logits
            + 0.0669677734375 * up_logits
>>>>>>> REPLACE