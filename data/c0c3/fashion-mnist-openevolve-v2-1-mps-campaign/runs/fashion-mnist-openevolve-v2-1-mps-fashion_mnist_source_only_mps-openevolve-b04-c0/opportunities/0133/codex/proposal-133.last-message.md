MECHANISM: Confidence-gated vertical-translation TTA

HYPOTHESIS: A low-weight symmetric one-pixel vertical TTA, used only when it preserves the stable prediction and increases that class’s calibrated probability, will retain 9,359 correct predictions while reducing cross-entropy below 0.18456672592163087.

INTENDED_EDIT: Add up/down translation logits during evaluation and selectively adopt their guarded 3.125% blend when it increases confidence in the unchanged stable class.

EVIDENCE: The horizontal TTA and argmax guard preserved accuracy through prior refinements, while the latest lower-weight probe worsened cross-entropy, indicating that further scalar weight tuning is locally exhausted and motivating an orthogonal guarded augmentation.

<<<<<<< SEARCH
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            corrected_logits,
        )
        return 1.226016 * ensemble_logits
=======
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            corrected_logits,
        )

        translated_images = torch.cat(
            (
                images.roll(shifts=1, dims=-2),
                images.roll(shifts=-1, dims=-2),
            ),
            dim=0,
        )
        translated_logits = self.classifier(
            self.features(translated_images)
        )
        down_logits, up_logits = translated_logits.chunk(2, dim=0)
        translation_refined_logits = (
            0.96875 * ensemble_logits
            + 0.015625 * down_logits
            + 0.015625 * up_logits
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
        return torch.where(
            use_translation,
            calibrated_translation_logits,
            calibrated_logits,
        )
>>>>>>> REPLACE