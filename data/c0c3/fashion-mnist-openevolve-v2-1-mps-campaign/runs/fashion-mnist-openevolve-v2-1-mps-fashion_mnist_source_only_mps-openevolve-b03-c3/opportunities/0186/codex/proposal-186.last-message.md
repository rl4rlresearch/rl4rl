MECHANISM: Arithmetic probability TTA ensemble

HYPOTHESIS: Averaging calibrated class probabilities instead of logits across the existing weighted views will exceed 9,284 correct predictions by limiting the effect of confidently wrong translated views.

INTENDED_EDIT: Preserve training, model parameters, TTA transforms, weights, and final temperature while replacing the geometric logit ensemble with an arithmetic probability mixture.

EVIDENCE: Logit-scale optimization saturated at 9,284 correct, while aligning training translations to TTA reduced accuracy to 9,262; this motivates a rank-changing evaluation-only ensemble that retains the validated augmentation weights.

<<<<<<< SEARCH
    def _flip_average(self, images: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images)
            + self._forward_once(images.flip(-1))
        )
=======
    def _flip_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images).softmax(dim=1)
            + self._forward_once(images.flip(-1)).softmax(dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = 0.3640625 * self._flip_average(images)
=======
        probabilities = 0.3640625 * self._flip_probabilities(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                logits = logits + weight * self._flip_average(shifted)
        return 1.167286 * logits
=======
                probabilities = (
                    probabilities
                    + weight * self._flip_probabilities(shifted)
                )
        return 1.167286 * probabilities.clamp_min(1.0e-8).log()
>>>>>>> REPLACE