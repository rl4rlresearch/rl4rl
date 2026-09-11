MECHANISM: Continuous refinement of accepted vertical-TTA predictions

HYPOTHESIS: Increasing only the accepted translation blend from 13.4375% to 13.44% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436292915344238.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72% to each vertical shift in the returned refined logits.

EVIDENCE: Decoupling selection and refinement at 13.4375% improved cross-entropy without changing correctness, and prior interpolation placed the continuous blend optimum near 13.44%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865625 * ensemble_logits
            + 0.0671875 * down_logits
            + 0.0671875 * up_logits
        )
=======
        translation_refined_logits = (
            0.8656 * ensemble_logits
            + 0.0672 * down_logits
            + 0.0672 * up_logits
        )
>>>>>>> REPLACE