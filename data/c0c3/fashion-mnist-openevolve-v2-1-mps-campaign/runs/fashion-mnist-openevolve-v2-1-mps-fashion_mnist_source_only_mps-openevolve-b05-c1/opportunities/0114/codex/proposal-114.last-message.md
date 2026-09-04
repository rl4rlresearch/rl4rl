MECHANISM: Conservative center-prior interpolation

HYPOTHESIS: A center weight of 1.875 will retain at least 9,360 correct predictions while reducing validation cross-entropy below 0.186017, strictly improving validation_score.

INTENDED_EDIT: Reduce centered original and flipped logit weights from 2.0 to 1.875 and renormalize the ten-view logit average.

EVIDENCE: Weight 2.0 achieved 9,360 correct, while weight 1.5 lowered cross-entropy to 0.185987 but lost three correct predictions; 1.875 conservatively moves toward the better-calibrated setting while minimizing classification-boundary changes.

<<<<<<< SEARCH
        logits = 2.0 * self._forward_once(views[0])
        logits = logits + 2.0 * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (2.0 * (len(views) + 1))
=======
        center_weight = 1.875
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
>>>>>>> REPLACE