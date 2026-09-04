MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing only the accepted translation blend from 13.4625% to 13.465% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628791809082.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.7325% to each vertical shift in the returned refined logits.

EVIDENCE: Every accepted-blend increase from 13.4375% through 13.4625% preserved correctness and reduced cross-entropy; the latest increase produced the largest recent reduction, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865375 * ensemble_logits
            + 0.0673125 * down_logits
            + 0.0673125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86535 * ensemble_logits
            + 0.067325 * down_logits
            + 0.067325 * up_logits
        )
>>>>>>> REPLACE