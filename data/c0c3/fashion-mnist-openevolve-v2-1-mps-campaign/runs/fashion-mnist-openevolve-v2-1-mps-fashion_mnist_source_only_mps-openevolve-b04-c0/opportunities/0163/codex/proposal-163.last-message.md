MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.45% to 13.4525% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436290550231935.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72625% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.45% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8655 * ensemble_logits
            + 0.06725 * down_logits
            + 0.06725 * up_logits
        )
=======
        translation_refined_logits = (
            0.865475 * ensemble_logits
            + 0.0672625 * down_logits
            + 0.0672625 * up_logits
        )
>>>>>>> REPLACE