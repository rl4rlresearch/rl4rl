MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.495% to 13.4975% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628242492676.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7475% to 6.74875%.

EVIDENCE: Every tested increase from 13.4375% through 13.495% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86505 * ensemble_logits
            + 0.067475 * down_logits
            + 0.067475 * up_logits
        )
=======
        translation_refined_logits = (
            0.865025 * ensemble_logits
            + 0.0674875 * down_logits
            + 0.0674875 * up_logits
        )
>>>>>>> REPLACE