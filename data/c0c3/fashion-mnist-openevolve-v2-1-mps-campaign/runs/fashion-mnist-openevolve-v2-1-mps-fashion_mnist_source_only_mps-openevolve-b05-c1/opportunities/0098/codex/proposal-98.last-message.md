MECHANISM: Confidence-preserving logit-space TTA consensus

HYPOTHESIS: Weighted logit averaging will exceed 9,345 correct predictions by preserving class-margin evidence across translated and flipped views instead of compressing each view independently through softmax.

INTENDED_EDIT: Replace probability-space test-time augmentation averaging with weighted raw-logit averaging while preserving the existing views, center weighting, and output temperature.

EVIDENCE: The raw-mean gated design achieved the best 9,345 correct, while several training-side representation and regularization changes regressed or timed out; changing only ensemble aggregation tests an orthogonal, parameter-free improvement without affecting training exposure.

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