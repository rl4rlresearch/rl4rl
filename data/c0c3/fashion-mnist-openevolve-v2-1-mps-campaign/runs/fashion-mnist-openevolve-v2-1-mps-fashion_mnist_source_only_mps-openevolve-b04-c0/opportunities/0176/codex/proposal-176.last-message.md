MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.4825% to 13.485% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628448486328.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74125% to 6.7425%.

EVIDENCE: Every tested increase from 13.4375% through 13.4825% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865175 * ensemble_logits
            + 0.0674125 * down_logits
            + 0.0674125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86515 * ensemble_logits
            + 0.067425 * down_logits
            + 0.067425 * up_logits
        )
>>>>>>> REPLACE