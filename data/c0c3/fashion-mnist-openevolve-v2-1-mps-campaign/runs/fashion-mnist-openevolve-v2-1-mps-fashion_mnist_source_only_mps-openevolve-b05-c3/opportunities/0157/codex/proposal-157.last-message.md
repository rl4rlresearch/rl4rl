MECHANISM: Pair-batched interior-weight probability TTA

HYPOTHESIS: A centered-view weight of 0.375 will exceed 9,208 correct predictions by refining the observed interior optimum between the inferior zero- and unit-weight endpoints, while paired inference will avoid the prior verification timeout.

INTENDED_EDIT: Batch each view with its horizontal flip in one forward pass, reduce both centered-view weights from 0.5 to 0.375, and normalize the unchanged eight unit-weight shifted views by the resulting total weight of 8.75.

EVIDENCE: Centered weights 0.0, 0.5, and 1.0 produced 9,206, 9,208, and 9,206 correct respectively; the previous 0.375 test timed out without performance evidence, so compute-equivalent paired inference makes that unresolved interpolation test more likely to complete.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.5
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.5,
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        paired_logits = self._forward_once(
            torch.cat((images, images.flip(-1)), dim=0)
        )
        centered, centered_flipped = F.softmax(
            paired_logits * 1.05, dim=1
        ).chunk(2, dim=0)
        probability_sum = centered.mul(0.375)
        probability_sum.add_(centered_flipped, alpha=0.375)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            paired_logits = self._forward_once(
                torch.cat((view, view.flip(-1)), dim=0)
            )
            view_probabilities, flipped_probabilities = F.softmax(
                paired_logits * 1.05, dim=1
            ).chunk(2, dim=0)
            probability_sum.add_(view_probabilities)
            probability_sum.add_(flipped_probabilities)

        return (probability_sum / 8.75).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE