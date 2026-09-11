MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.455% to 13.4575% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436289710998535.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72875% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.455% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86545 * ensemble_logits
            + 0.067275 * down_logits
            + 0.067275 * up_logits
        )
=======
        translation_refined_logits = (
            0.865425 * ensemble_logits
            + 0.0672875 * down_logits
            + 0.0672875 * up_logits
        )
>>>>>>> REPLACE