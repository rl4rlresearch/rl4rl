MECHANISM: Horizontally symmetric probability-space TTA

HYPOTHESIS: Splitting the qualified horizontal-view weight equally between left- and right-shifted views will exceed 9,210 correct predictions by removing directional inference bias while preserving the best ensemble’s total horizontal, centered, and vertical weight.

INTENDED_EDIT: Restore the qualified training sampler and calibration, then use ten inference passes with half-weighted left/right shift pairs, full-weighted vertical pairs, and unchanged total normalization.

EVIDENCE: Reference Design 1 achieved 9,210 correct with an asymmetric eight-pass ensemble, while the equal-weight ten-pass current design achieved 9,206; preserving the qualified view-group weights isolates horizontal symmetry without the current design’s confounded reweighting.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
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

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.375,
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        vertical_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
        )
        for view in vertical_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        horizontal_views = (
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in horizontal_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(
                F.softmax(view_logits * 1.10, dim=1),
                alpha=0.5,
            )
            probability_sum.add_(
                F.softmax(flipped_logits * 1.10, dim=1),
                alpha=0.5,
            )

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        directions = torch.randint(
            0, 6, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE