MECHANISM: Fine-grained center-prior logit-space TTA

HYPOTHESIS: Increasing centered-view weight from 2.0 to 2.125 will exceed 9,360 correct predictions by modestly strengthening the useful center prior without the overemphasis observed at weight 3.0.

INTENDED_EDIT: Weight the centered original and flipped logits by 2.125 and renormalize the ten-view logit average.

EVIDENCE: Center weight 2.0 achieved 9,360 correct, while weights 1.5 and 3.0 achieved only 9,357 and 9,355; this brackets the best setting and motivates a conservative local refinement near 2.0.

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
        center_weight = 2.125
        logits = center_weight * self._forward_once(views[0])
        logits = logits + center_weight * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (2.0 * (len(views) - 1 + center_weight))
>>>>>>> REPLACE