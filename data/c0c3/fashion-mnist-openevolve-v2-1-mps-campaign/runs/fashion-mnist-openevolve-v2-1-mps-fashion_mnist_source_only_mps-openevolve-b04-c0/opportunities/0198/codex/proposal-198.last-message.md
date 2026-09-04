MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5375% to 13.54% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436275100708008.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76875% to 6.77%.

EVIDENCE: Every tested increase from 13.4375% through 13.5375% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864625 * ensemble_logits
            + 0.0676875 * down_logits
            + 0.0676875 * up_logits
        )
=======
        translation_refined_logits = (
            0.8646 * ensemble_logits
            + 0.0677 * down_logits
            + 0.0677 * up_logits
        )
>>>>>>> REPLACE