MECHANISM: Reduced centered-view influence in symmetric probability-space TTA

HYPOTHESIS: Reducing each centered-view weight from 0.375 to 0.25 will exceed 9,215 correct predictions by favoring the translation-robust symmetric shifted views.

INTENDED_EDIT: Install the verified symmetric ten-pass ensemble, reduce both centered-view weights to 0.25, normalize by 6.5, and restore 1.178× calibration.

EVIDENCE: Symmetric horizontal weighting improved validation_correct from 9,210 to 9,215; increasing centered weights to 0.5 then lost one correct prediction, motivating the opposite centered-weight perturbation.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
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
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
=======
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.25
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.25,
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

        return (probability_sum / 6.5).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE