MECHANISM: Center-prior logit-space TTA

HYPOTHESIS: Increasing each centered-view weight from 2 to 3 will exceed 9,360 correct predictions by reducing the influence of less reliable translated views while preserving flip consensus.

INTENDED_EDIT: Increase the centered original and flipped logits to weight 3 and renormalize the ten-view logit average.

EVIDENCE: Weighted raw-logit aggregation improved validation correct from 9,358 to 9,360; the resulting best design already privileges centered views, motivating a focused test of a stronger center prior without changing training, parameters, or inference passes.

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
        logits = 3.0 * self._forward_once(views[0])
        logits = logits + 3.0 * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (2.0 * (len(views) + 2))
>>>>>>> REPLACE