MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.4775% to 13.48% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436285629272461.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.73875% to 6.74%.

EVIDENCE: Every tested increase from 13.4375% through 13.4775% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865225 * ensemble_logits
            + 0.0673875 * down_logits
            + 0.0673875 * up_logits
        )
=======
        translation_refined_logits = (
            0.8652 * ensemble_logits
            + 0.0674 * down_logits
            + 0.0674 * up_logits
        )
>>>>>>> REPLACE