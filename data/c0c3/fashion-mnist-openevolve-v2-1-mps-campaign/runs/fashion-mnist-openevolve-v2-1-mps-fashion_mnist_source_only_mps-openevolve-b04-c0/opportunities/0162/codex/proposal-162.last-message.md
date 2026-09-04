MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.4475% to 13.45% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843629119873047.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.725% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.4475% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865525 * ensemble_logits
            + 0.0672375 * down_logits
            + 0.0672375 * up_logits
        )
=======
        translation_refined_logits = (
            0.8655 * ensemble_logits
            + 0.06725 * down_logits
            + 0.06725 * up_logits
        )
>>>>>>> REPLACE