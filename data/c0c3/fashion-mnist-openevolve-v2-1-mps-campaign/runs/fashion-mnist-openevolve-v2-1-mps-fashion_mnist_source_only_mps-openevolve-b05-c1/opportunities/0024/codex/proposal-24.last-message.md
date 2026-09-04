MECHANISM: Probability-space test-time augmentation fusion

HYPOTHESIS: Averaging per-view class probabilities will exceed 9,240 correct predictions, or tie while lowering cross-entropy, by preventing an overconfident translated or flipped view from dominating the ensemble’s decision.

INTENDED_EDIT: Preserve the verified model and training procedure, but replace evaluation-time logit averaging with weighted probability averaging and return the resulting log-probabilities.

EVIDENCE: The strongest design relies on center/cardinal translated-view inference, while the successful 3.3e-3 learning rate slightly worsened cross-entropy; bounded probability fusion directly targets ensemble robustness and calibration without changing training time, learned parameters, or optimizer behavior.

<<<<<<< SEARCH
        logits = 2.0 * self._forward_once(views[0])
        logits = logits + 2.0 * self._forward_once(views[0].flip(-1))
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        return logits / (2.0 * (len(views) + 1))
=======
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
        return probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE