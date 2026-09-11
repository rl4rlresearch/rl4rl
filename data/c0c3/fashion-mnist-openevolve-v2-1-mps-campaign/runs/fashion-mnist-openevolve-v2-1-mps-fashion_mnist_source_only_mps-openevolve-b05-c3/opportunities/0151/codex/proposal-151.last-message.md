MECHANISM: Half-weight centered TTA with stronger argmax-invariant calibration

HYPOTHESIS: Restoring the qualified 0.5-weight centered views will recover 9,208 correct predictions, while increasing post-ensemble sharpening from 1.10× to 1.15× will preserve those predictions and reduce cross-entropy below 0.224548.

INTENDED_EDIT: Add the original and horizontally flipped views at half weight alongside the eight shifted views, normalize by 9.0, and apply 1.15× final sharpening.

EVIDENCE: Reference Design 1 verified half-weight centered views at 9,208 correct and 0.224548 cross-entropy, outperforming the current shift-only design’s 9,206 correct. Reference Designs 2 and 3 showed that increasing final sharpening from 1.05× to 1.10× preserved correctness while lowering cross-entropy from 0.227083 to 0.224459.

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

        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE