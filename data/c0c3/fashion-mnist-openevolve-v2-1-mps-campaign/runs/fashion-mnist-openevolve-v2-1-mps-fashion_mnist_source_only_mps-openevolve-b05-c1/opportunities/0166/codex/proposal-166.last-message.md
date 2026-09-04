MECHANISM: Low-weight diagonal translation marginalization

HYPOTHESIS: Redistributing one eighth of the off-center TTA weight to diagonal one-pixel translations will exceed 9,360 correct predictions by reducing two-axis pooling-phase sensitivity while preserving the verified logit-space aggregation and calibration.

INTENDED_EDIT: Extend evaluation TTA with four diagonal translations at 0.125 weight each and reduce cardinal-view weights to 0.875, keeping the total off-center weight and center weight unchanged.

EVIDENCE: Learned anti-aliased downsampling targeted phase sensitivity but timed out, while probability-space TTA lost one correct prediction; this tests translation stability without changing the verified architecture, training procedure, logit aggregation, or total ensemble weight.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        center_weight = 1.81732177734375
        logits = center_weight * self._forward_once(views[0])
        logits = logits + center_weight * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )
        return 1.0671112 * logits
=======
        cardinal_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        diagonal_views = (
            padded[:, :, 0:28, 0:28],
            padded[:, :, 0:28, 2:30],
            padded[:, :, 2:30, 0:28],
            padded[:, :, 2:30, 2:30],
        )
        center_weight = 1.81732177734375
        cardinal_weight = 0.875
        diagonal_weight = 0.125
        logits = center_weight * self._forward_once(images)
        logits = logits + center_weight * self._forward_once(
            images.flip(-1)
        )
        for view in cardinal_views:
            logits = logits + cardinal_weight * self._forward_once(view)
            logits = logits + cardinal_weight * self._forward_once(
                view.flip(-1)
            )
        for view in diagonal_views:
            logits = logits + diagonal_weight * self._forward_once(view)
            logits = logits + diagonal_weight * self._forward_once(
                view.flip(-1)
            )
        logits = logits / (
            2.0 * center_weight
            + 2.0 * (
                len(cardinal_views) * cardinal_weight
                + len(diagonal_views) * diagonal_weight
            )
        )
        return 1.0671112 * logits
>>>>>>> REPLACE