MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.4575% to 13.46% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436289443969728.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.73% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.4575% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865425 * ensemble_logits
            + 0.0672875 * down_logits
            + 0.0672875 * up_logits
        )
=======
        translation_refined_logits = (
            0.8654 * ensemble_logits
            + 0.0673 * down_logits
            + 0.0673 * up_logits
        )
>>>>>>> REPLACE