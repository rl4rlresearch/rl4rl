MECHANISM: Conservative per-view temperature softening with calibrated fusion

HYPOTHESIS: Scaling each view’s logits by 0.95 before probability averaging will exceed 9,249 correct predictions by further reducing outlier-view dominance; if correctness ties, 1.1 post-fusion sharpening will lower cross-entropy below 0.216368.

INTENDED_EDIT: Mildly soften every test-time view distribution before fusion, then sharpen only the fused log-probabilities without changing their argmax.

EVIDENCE: Probability averaging improved correctness from 9,240 to 9,249, while 1.1 post-fusion sharpening retained all 9,249 predictions and reduced cross-entropy to 0.215002; 0.95 is a conservative intermediate test after the unmeasured 0.9-softening verification timed out.

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
        return probabilities.clamp_min(1e-8).log()
=======
        view_logit_scale = 0.95
        probabilities = 2.0 * F.softmax(
            view_logit_scale * self._forward_once(views[0]), dim=1
        )
        probabilities = probabilities + 2.0 * F.softmax(
            view_logit_scale * self._forward_once(views[0].flip(-1)), dim=1
        )
        for view in views[1:]:
            probabilities = probabilities + F.softmax(
                view_logit_scale * self._forward_once(view), dim=1
            )
            probabilities = probabilities + F.softmax(
                view_logit_scale * self._forward_once(view.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * (len(views) + 1))
        return 1.1 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE