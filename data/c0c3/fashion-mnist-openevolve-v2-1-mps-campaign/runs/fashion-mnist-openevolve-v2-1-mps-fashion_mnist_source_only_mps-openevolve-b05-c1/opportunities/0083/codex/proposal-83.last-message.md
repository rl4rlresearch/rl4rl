MECHANISM: Softmax-free geometric test-time aggregation

HYPOTHESIS: Weighted logit averaging will exceed 9,345 correct predictions by favoring evidence consistent across augmented views while avoiding the evaluation overhead of computing ten softmax distributions.

INTENDED_EDIT: Replace arithmetic probability averaging with its computationally cheaper weighted-logit counterpart, preserving view weights and calibration scaling.

EVIDENCE: Geometric aggregation was previously motivated by the 9,345-correct raw-mean-gating result but timed out; weighted logit averaging gives the same class decisions as weighted log-softmax averaging while eliminating per-view softmax operations.

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
        logits = logits + 2.0 * self._forward_once(views[0].flip(-1))
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (2.0 * (len(views) + 1))
        return 1.104 * logits
>>>>>>> REPLACE