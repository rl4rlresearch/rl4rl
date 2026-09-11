MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.445% to 13.4475% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436291732788085.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72375% to each vertical shift in the returned refined logits.

EVIDENCE: Increasing the decoupled accepted blend from 13.4375% through 13.445% repeatedly preserved correctness and reduced cross-entropy; another equal-sized probe most directly tests whether the continuous optimum lies higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86555 * ensemble_logits
            + 0.067225 * down_logits
            + 0.067225 * up_logits
        )
=======
        translation_refined_logits = (
            0.865525 * ensemble_logits
            + 0.0672375 * down_logits
            + 0.0672375 * up_logits
        )
>>>>>>> REPLACE