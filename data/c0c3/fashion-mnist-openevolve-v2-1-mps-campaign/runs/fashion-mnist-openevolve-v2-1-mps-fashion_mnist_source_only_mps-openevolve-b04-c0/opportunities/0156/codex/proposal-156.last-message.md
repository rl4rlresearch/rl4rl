MECHANISM: One-sided bisection of the vertical-TTA gating boundary

HYPOTHESIS: A 13.397434234619140625% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300659179689 while remaining below the degradation observed at 13.39746856689453125%.

INTENDED_EDIT: Increase total translated-view weight from 13.39739990234375% to 13.397434234619140625%, assigning 6.6987171173095703125% to each vertical shift.

EVIDENCE: The 13.39739990234375% blend is the best verified design, while 13.39746856689453125% sharply worsened cross-entropy; their midpoint is the most informative next probe of the apparent confidence-gating boundary.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8660260009765625 * ensemble_logits
            + 0.06698699951171875 * down_logits
            + 0.06698699951171875 * up_logits
        )
=======
        translation_refined_logits = (
            0.86602565765380859375 * ensemble_logits
            + 0.066987171173095703125 * down_logits
            + 0.066987171173095703125 * up_logits
        )
>>>>>>> REPLACE