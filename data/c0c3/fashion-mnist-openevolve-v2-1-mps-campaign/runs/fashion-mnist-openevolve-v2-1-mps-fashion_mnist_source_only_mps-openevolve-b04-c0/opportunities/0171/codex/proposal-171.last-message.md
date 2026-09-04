MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.47% to 13.4725% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436286964416504.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.735% to 6.73625%.

EVIDENCE: Every tested increase from 13.4375% through 13.47% preserved correctness and reduced cross-entropy; an equal-sized one-sided probe most directly tests whether the optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8653 * ensemble_logits
            + 0.06735 * down_logits
            + 0.06735 * up_logits
        )
=======
        translation_refined_logits = (
            0.865275 * ensemble_logits
            + 0.0673625 * down_logits
            + 0.0673625 * up_logits
        )
>>>>>>> REPLACE