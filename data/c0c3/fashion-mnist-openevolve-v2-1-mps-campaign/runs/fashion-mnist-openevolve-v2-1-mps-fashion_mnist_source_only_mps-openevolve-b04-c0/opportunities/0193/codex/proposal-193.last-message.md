MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.525% to 13.5275% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627700805664.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7625% to 6.76375%.

EVIDENCE: Every tested increase from 13.4375% through 13.525% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86475 * ensemble_logits
            + 0.067625 * down_logits
            + 0.067625 * up_logits
        )
=======
        translation_refined_logits = (
            0.864725 * ensemble_logits
            + 0.0676375 * down_logits
            + 0.0676375 * up_logits
        )
>>>>>>> REPLACE