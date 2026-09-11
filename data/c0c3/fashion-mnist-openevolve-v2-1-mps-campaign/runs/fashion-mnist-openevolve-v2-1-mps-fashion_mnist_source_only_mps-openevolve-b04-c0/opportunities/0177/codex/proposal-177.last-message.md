MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.485% to 13.4875% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436283950805665.

INTENDED_EDIT: Keep the 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7425% to 6.74375%.

EVIDENCE: Every tested increase from 13.4375% through 13.485% preserved correctness and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86515 * ensemble_logits
            + 0.067425 * down_logits
            + 0.067425 * up_logits
        )
=======
        translation_refined_logits = (
            0.865125 * ensemble_logits
            + 0.0674375 * down_logits
            + 0.0674375 * up_logits
        )
>>>>>>> REPLACE