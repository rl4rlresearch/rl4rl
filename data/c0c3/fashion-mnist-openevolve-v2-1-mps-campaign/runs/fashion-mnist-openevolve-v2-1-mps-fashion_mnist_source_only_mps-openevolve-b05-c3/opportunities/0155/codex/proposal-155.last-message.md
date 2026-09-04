MECHANISM: Half-weight centered TTA with conservative temperature refinement

HYPOTHESIS: Restoring the verified half-weight centered ensemble will recover 9,208 correct predictions, while a 1.125× final multiplier will preserve its argmax decisions and reduce cross-entropy below 0.224548.

INTENDED_EDIT: Add centered and horizontally flipped predictions at 0.5 weight, retain all eight unit-weight shifted predictions, normalize by 9.0, and sharpen the resulting logits by 1.125×.

EVIDENCE: Reference Design 1 achieved the best verified correctness, 9,208, with half-weight centered views; increasing final sharpening from 1.05× to 1.10× previously preserved correctness and reduced cross-entropy, motivating a conservative midpoint toward the unresolved 1.15× test.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )

        first_view = shifted_views[0]
        probability_sum = F.softmax(
            self._forward_once(first_view) * 1.05,
            dim=1,
        )
        probability_sum.add_(
            F.softmax(
                self._forward_once(first_view.flip(-1)) * 1.05,
                dim=1,
            )
        )
        for view in shifted_views[1:]:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 8.0).clamp_min(1e-8).log().mul_(1.10)
=======
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

        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.125)
>>>>>>> REPLACE