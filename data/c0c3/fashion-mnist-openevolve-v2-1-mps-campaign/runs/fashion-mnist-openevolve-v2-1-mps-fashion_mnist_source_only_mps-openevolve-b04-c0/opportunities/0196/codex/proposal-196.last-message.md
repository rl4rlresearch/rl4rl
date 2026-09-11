MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5325% to 13.535% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627586364746.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76625% to 6.7675%.

EVIDENCE: Every tested increase from 13.4375% through 13.5325% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864675 * ensemble_logits
            + 0.0676625 * down_logits
            + 0.0676625 * up_logits
        )
=======
        translation_refined_logits = (
            0.86465 * ensemble_logits
            + 0.067675 * down_logits
            + 0.067675 * up_logits
        )
>>>>>>> REPLACE