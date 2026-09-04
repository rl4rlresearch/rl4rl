MECHANISM: Bisection refinement of the confidence-gated vertical-TTA boundary

HYPOTHESIS: A 13.395751953125% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436301498413085 while remaining below the degradation observed at 13.39794921875%.

INTENDED_EDIT: Increase total translated-view weight from 13.3935546875% to 13.395751953125%, assigning 6.6978759765625% to each vertical shift.

EVIDENCE: The 13.3935546875% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.

<<<<<<< SEARCH
            0.866064453125 * ensemble_logits
            + 0.0669677734375 * down_logits
            + 0.0669677734375 * up_logits
=======
            0.86604248046875 * ensemble_logits
            + 0.066978759765625 * down_logits
            + 0.066978759765625 * up_logits
>>>>>>> REPLACE