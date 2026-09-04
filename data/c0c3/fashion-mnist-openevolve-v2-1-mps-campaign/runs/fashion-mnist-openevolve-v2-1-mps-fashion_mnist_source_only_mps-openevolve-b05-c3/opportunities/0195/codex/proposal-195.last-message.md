MECHANISM: Horizontally symmetric probability-space test-time augmentation

HYPOTHESIS: The verified symmetric ten-pass ensemble will increase validation_correct from 9,210 to at least 9,215 and achieve cross-entropy near 0.222402.

INTENDED_EDIT: Split the existing horizontal-shift weight equally between left and right views while preserving total ensemble weight, and restore the verified 1.178× calibration.

EVIDENCE: Reference Design 3 made this inference-only change and achieved the best available result: 9,215 correct predictions and 0.2224023125 cross-entropy.

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
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

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.1875)
=======
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