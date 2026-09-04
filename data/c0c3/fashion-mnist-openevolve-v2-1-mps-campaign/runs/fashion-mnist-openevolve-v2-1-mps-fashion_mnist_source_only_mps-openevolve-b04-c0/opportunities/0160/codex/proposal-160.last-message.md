MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.4425% to 13.445% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436291847229003.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.7225% to each vertical shift in the returned refined logits.

EVIDENCE: Each increase from 13.4375% through 13.44% to 13.4425% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe is the most informative test of whether the continuous optimum lies higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865575 * ensemble_logits
            + 0.0672125 * down_logits
            + 0.0672125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86555 * ensemble_logits
            + 0.067225 * down_logits
            + 0.067225 * up_logits
        )
>>>>>>> REPLACE