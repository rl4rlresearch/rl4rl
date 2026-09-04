MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.46% to 13.4625% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436288795471192.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.73125% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.46% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8654 * ensemble_logits
            + 0.0673 * down_logits
            + 0.0673 * up_logits
        )
=======
        translation_refined_logits = (
            0.865375 * ensemble_logits
            + 0.0673125 * down_logits
            + 0.0673125 * up_logits
        )
>>>>>>> REPLACE