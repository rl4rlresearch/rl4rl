MECHANISM: Anisotropic cardinal-view logit averaging

HYPOTHESIS: Redistributing one eighth of vertical-view weight to horizontal translations will exceed 9,360 correct predictions by preserving class-bearing vertical alignment while retaining horizontal translation robustness.

INTENDED_EDIT: Keep the existing ten views, center weight, total ensemble weight, logit-space aggregation, and calibration, but weight vertical shifts at 0.875 and horizontal shifts at 1.125.

EVIDENCE: Diagonal translation augmentation reduced validation_correct from 9,360 to 9,340, indicating that translation invariance is not uniformly beneficial; probability-space TTA also underperformed, motivating a targeted weight redistribution within the verified logit-space ensemble.

<<<<<<< SEARCH
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
=======
        center_weight = 1.81732177734375
        logits = center_weight * self._forward_once(views[0])
        logits = logits + center_weight * self._forward_once(
            views[0].flip(-1)
        )
        view_weights = (0.875, 0.875, 1.125, 1.125)
        for view, view_weight in zip(views[1:], view_weights):
            logits = logits + view_weight * self._forward_once(view)
            logits = logits + view_weight * self._forward_once(
                view.flip(-1)
            )
        logits = logits / (
            2.0 * center_weight + 2.0 * sum(view_weights)
        )
>>>>>>> REPLACE