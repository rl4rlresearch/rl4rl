MECHANISM: Confidence-preserving logit-space TTA consensus

HYPOTHESIS: Weighted logit averaging will exceed 9,358 correct predictions by reinforcing class-margin agreement across translated and flipped views without altering training or parameter count.

INTENDED_EDIT: Replace probability-space TTA averaging with weighted raw-logit averaging while preserving the views, center weighting, normalization, and output temperature.

EVIDENCE: The 9,358-correct design still averages per-view probabilities; the prior weighted-logit attempt timed out without accuracy evidence, so this orthogonal, computation-neutral aggregation change remains unresolved.

<<<<<<< SEARCH
        probabilities = 2.0 * F.softmax(
            self._forward_once(views[0]), dim=1
        )
        probabilities = probabilities + 2.0 * F.softmax(
            self._forward_once(views[0].flip(-1)), dim=1
        )
        for view in views[1:]:
            probabilities = probabilities + F.softmax(
                self._forward_once(view), dim=1
            )
            probabilities = probabilities + F.softmax(
                self._forward_once(view.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * (len(views) + 1))
        return 1.104 * probabilities.clamp_min(1e-8).log()
=======
        logits = 2.0 * self._forward_once(views[0])
        logits = logits + 2.0 * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (2.0 * (len(views) + 1))
        return 1.104 * logits
>>>>>>> REPLACE