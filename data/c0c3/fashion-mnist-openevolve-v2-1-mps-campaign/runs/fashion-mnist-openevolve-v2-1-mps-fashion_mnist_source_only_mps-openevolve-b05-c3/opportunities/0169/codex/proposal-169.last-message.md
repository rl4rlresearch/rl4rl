MECHANISM: Stronger confidence-weighted probability TTA

HYPOTHESIS: Increasing per-view logit sharpening from 1.05× to 1.10× will exceed 9,210 correct predictions by giving the most confident centered and shifted views greater influence without changing the qualified training trajectory or inference workload.

INTENDED_EDIT: Increase only the pre-softmax temperature multiplier for every TTA view, retaining the verified 1/3-centered training sampler, view weights, and 1.15× final calibration.

EVIDENCE: The current 1.05× per-view ensemble is the best qualified 9,210-correct design, while prior evidence specifically identified per-view sharpening as the runtime-neutral mechanism capable of changing ensemble argmax decisions; unlike final sharpening, this isolates a correctness-relevant inference adjustment.

<<<<<<< SEARCH
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
>>>>>>> REPLACE