MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.465% to 13.4675% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436287574768068.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.73375% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.465% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86535 * ensemble_logits
            + 0.067325 * down_logits
            + 0.067325 * up_logits
        )
=======
        translation_refined_logits = (
            0.865325 * ensemble_logits
            + 0.0673375 * down_logits
            + 0.0673375 * up_logits
        )
>>>>>>> REPLACE