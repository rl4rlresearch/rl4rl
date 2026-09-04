MECHANISM: Flip-symmetric eight-pass centered TTA

HYPOTHESIS: Replacing one horizontally redundant shift pair with centered views while preserving the best verified 1:8 centered-to-shifted weight ratio will reach at least 9,208 correct predictions without increasing inference work.

INTENDED_EDIT: Evaluate centered and flipped-centered views at 0.375 weight each, retain three unit-weight shift pairs, omit one horizontal shift pair, and normalize the eight-pass ensemble by 6.75.

EVIDENCE: Half-weight centered views with eight shifted views achieved the best verified result of 9,208 correct at a 1:8 centered-to-shifted weight ratio, whereas shift-only inference achieved 9,206. Prior ten-view attempts repeatedly failed verification, motivating the same ratio within the current eight-forward runtime budget.

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

        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.375,
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE