MECHANISM: Boundary-seeking per-view temperature interpolation

HYPOTHESIS: A 1.125× per-view multiplier will restore 9,210 correct predictions while lowering cross-entropy below the 0.2230063 achieved at 1.10×.

INTENDED_EDIT: Replace the current 1.15× per-view sharpening with the unverified 1.125× midpoint, retaining the qualified 1/3-centered training sampler and 1.15× final calibration.

EVIDENCE: Per-view sharpening retained 9,210 correct at 1.10× but lost one prediction at 1.15×; the prior 1.125× attempt timed out, so the midpoint remains the most direct unresolved boundary test.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.15, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.15,
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
            probability_sum.add_(F.softmax(view_logits * 1.15, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.15, dim=1))
=======
        probability_sum = F.softmax(logits * 1.125, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.125,
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
            probability_sum.add_(F.softmax(view_logits * 1.125, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.125, dim=1))
>>>>>>> REPLACE