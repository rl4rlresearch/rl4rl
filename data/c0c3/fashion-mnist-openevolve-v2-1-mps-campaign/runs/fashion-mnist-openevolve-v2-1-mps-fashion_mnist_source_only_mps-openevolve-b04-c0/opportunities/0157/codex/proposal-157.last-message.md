MECHANISM: Decoupled translation selection and refinement weights

HYPOTHESIS: Freezing the translation gate at the verified 13.39739990234375% blend while applying a 13.4375% blend only to accepted predictions will preserve 9,359 correct predictions and reduce cross-entropy below 0.18436300659179689.

INTENDED_EDIT: Use the current best blend for confidence gating, then return a separately refined 13.4375% vertical-translation blend with an explicit prediction-stability guard.

EVIDENCE: Cross-entropy improved through 13.39739990234375%, while slightly larger weights caused an abrupt regression attributed to the discrete confidence gate; earlier interpolation placed the continuous blend optimum near 13.44%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8660260009765625 * ensemble_logits
            + 0.06698699951171875 * down_logits
            + 0.06698699951171875 * up_logits
        )

        calibrated_logits = 1.226016 * ensemble_logits
        calibrated_translation_logits = (
            1.226016 * translation_refined_logits
        )
        translation_unchanged = (
            translation_refined_logits.argmax(dim=1, keepdim=True)
            == stable_prediction
        )
        stable_log_probability = F.log_softmax(
            calibrated_logits,
            dim=1,
        ).gather(1, stable_prediction)
        translation_log_probability = F.log_softmax(
            calibrated_translation_logits,
            dim=1,
        ).gather(1, stable_prediction)
        use_translation = (
            translation_unchanged
            & (translation_log_probability > stable_log_probability)
        )
=======
        selection_translation_logits = (
            0.8660260009765625 * ensemble_logits
            + 0.06698699951171875 * down_logits
            + 0.06698699951171875 * up_logits
        )
        translation_refined_logits = (
            0.865625 * ensemble_logits
            + 0.0671875 * down_logits
            + 0.0671875 * up_logits
        )

        calibrated_logits = 1.226016 * ensemble_logits
        calibrated_selection_translation_logits = (
            1.226016 * selection_translation_logits
        )
        calibrated_translation_logits = (
            1.226016 * translation_refined_logits
        )
        selection_unchanged = (
            selection_translation_logits.argmax(dim=1, keepdim=True)
            == stable_prediction
        )
        translation_unchanged = (
            translation_refined_logits.argmax(dim=1, keepdim=True)
            == stable_prediction
        )
        stable_log_probability = F.log_softmax(
            calibrated_logits,
            dim=1,
        ).gather(1, stable_prediction)
        translation_log_probability = F.log_softmax(
            calibrated_selection_translation_logits,
            dim=1,
        ).gather(1, stable_prediction)
        use_translation = (
            selection_unchanged
            & translation_unchanged
            & (translation_log_probability > stable_log_probability)
        )
>>>>>>> REPLACE