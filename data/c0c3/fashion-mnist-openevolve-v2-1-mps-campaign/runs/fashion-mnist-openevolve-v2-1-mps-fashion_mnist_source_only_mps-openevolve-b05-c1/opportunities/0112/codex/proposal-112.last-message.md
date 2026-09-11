MECHANISM: Reduced center-prior logit-space TTA

HYPOTHESIS: Reducing centered-view weight from 2.0 to 1.5 will exceed 9,360 correct predictions by increasing translated-view influence without discarding the useful center prior.

INTENDED_EDIT: Weight the centered original and flipped logits by 1.5 and renormalize the ten-view logit average.

EVIDENCE: Increasing centered-view weight from 2 to 3 reduced validation correct from 9,360 to 9,355, indicating that stronger center emphasis is harmful and motivating a measured move in the opposite direction.

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
        logits = 1.5 * self._forward_once(views[0])
        logits = logits + 1.5 * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (2.0 * len(views) + 1.0)
>>>>>>> REPLACE