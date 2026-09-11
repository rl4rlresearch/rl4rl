MECHANISM: Mixture-of-views probability ensembling

HYPOTHESIS: Averaging calibrated class probabilities across the ten evaluation views will exceed 9,360 correct predictions by preventing one unusually confident translated view from dominating the geometric, logit-space ensemble.

INTENDED_EDIT: Preserve the verified architecture, training procedure, center weighting, and temperature while replacing evaluation-time weighted logit averaging with weighted probability averaging followed by conversion back to logits.

EVIDENCE: Temperature refinement plateaued at 9,360 correct, while class-attention and widened-head changes fell to 9,337 and 9,326; this motivates preserving the learned classifier and testing the previously unchanged TTA aggregation rule.

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
        return 1.0671112 * logits
=======
        center_weight = 1.81732177734375
        probabilities = center_weight * self._forward_once(
            views[0]
        ).softmax(dim=1)
        probabilities = probabilities + center_weight * self._forward_once(
            views[0].flip(-1)
        ).softmax(dim=1)
        for view in views[1:]:
            probabilities = probabilities + self._forward_once(
                view
            ).softmax(dim=1)
            probabilities = probabilities + self._forward_once(
                view.flip(-1)
            ).softmax(dim=1)
        probabilities = probabilities / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )
        return 1.0671112 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE