MECHANISM: Probability-space transformation marginalization

HYPOTHESIS: Restoring the 9,284-correct pooling learning rate and averaging TTA predictions in probability space will exceed 9,284 correct by preventing an overconfident shifted or flipped view from disproportionately controlling the ensemble.

INTENDED_EDIT: Restore full-rate optimization for both pooling gates and replace geometric logit averaging across the 50 evaluation views with an arithmetic probability ensemble converted back to calibrated logits.

EVIDENCE: Independent 90%-max pooling at full learning rate achieved the best result of 9,284 correct, while half-rate gates fell to 9,281 and architectural additions regressed or timed out; this preserves the best training design and targets the still-untested inference aggregation.

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
        logits = 0.36 * self._flip_average(images)
=======
        probabilities = 0.36 * self._flip_probabilities(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                logits = logits + weight * self._flip_average(shifted)
        return 1.253 * logits
=======
                probabilities = (
                    probabilities
                    + weight * self._flip_probabilities(shifted)
                )
        return 1.253 * probabilities.clamp_min(1.0e-7).log()
>>>>>>> REPLACE

<<<<<<< SEARCH
            {"params": pooling, "weight_decay": 0.0, "lr_scale": 0.5},
=======
            {"params": pooling, "weight_decay": 0.0, "lr_scale": 1.0},
>>>>>>> REPLACE