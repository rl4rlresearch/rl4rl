MECHANISM: Horizontally symmetric probability-space test-time augmentation

HYPOTHESIS: Splitting the horizontal-shift contribution equally between left- and right-shifted views will increase validation_correct from 9,210 to at least 9,215 while retaining cross-entropy near 0.222402.

INTENDED_EDIT: Replace the asymmetric six-shift-pass ensemble with the verified symmetric eight-shift-pass ensemble, preserving centered, vertical, total horizontal weights, normalization, and calibration.

EVIDENCE: Reference Design 3 made this isolated change and achieved the best available result: 9,215 correct predictions and 0.2224023125 cross-entropy.

<<<<<<< SEARCH
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
=======
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