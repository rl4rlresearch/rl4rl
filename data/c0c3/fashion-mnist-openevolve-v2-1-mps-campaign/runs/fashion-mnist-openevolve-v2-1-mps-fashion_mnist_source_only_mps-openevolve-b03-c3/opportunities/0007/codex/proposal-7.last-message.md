MECHANISM: Probability-space translation-orbit mixture

HYPOTHESIS: Replacing logit averaging with a weighted probability mixture over the proven 5×5 translation-and-flip ensemble will exceed 9,051 correct predictions by preventing weak radius-two views from disproportionately suppressing confident class probabilities.

INTENDED_EDIT: Add the best verified 5×5 translation ensemble and combine its flip-paired predictions as probabilities, returning normalized log-probabilities for validation.

EVIDENCE: The weighted 5×5 logit ensemble reached 9,051 correct, improving on both the 9,042-correct 3×3 ensemble and the 9,009-correct flip-only model; its slightly higher cross-entropy than the 3×3 result suggests aggregation, rather than view coverage, is the next useful variable.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
    def _flip_probability_average(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images).softmax(dim=-1)
            + self._forward_once(images.flip(-1)).softmax(dim=-1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        probabilities = 0.36 * self._flip_probability_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
                probabilities = probabilities + (
                    weight * self._flip_probability_average(shifted)
                )

        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
>>>>>>> REPLACE