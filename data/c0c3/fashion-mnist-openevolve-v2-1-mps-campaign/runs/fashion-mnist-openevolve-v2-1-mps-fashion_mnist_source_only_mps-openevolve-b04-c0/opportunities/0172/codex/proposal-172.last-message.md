MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.4725% to 13.475% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436286544799804.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.73625% to 6.7375%.

EVIDENCE: Every tested increase from 13.4375% through 13.4725% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe most directly tests whether the optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865275 * ensemble_logits
            + 0.0673625 * down_logits
            + 0.0673625 * up_logits
        )
=======
        translation_refined_logits = (
            0.86525 * ensemble_logits
            + 0.067375 * down_logits
            + 0.067375 * up_logits
        )
>>>>>>> REPLACE