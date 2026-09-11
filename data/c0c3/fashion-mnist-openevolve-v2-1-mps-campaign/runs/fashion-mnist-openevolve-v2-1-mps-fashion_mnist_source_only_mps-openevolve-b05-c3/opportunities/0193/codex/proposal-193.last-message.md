MECHANISM: Centered-view weight bracketing in symmetric probability-space TTA

HYPOTHESIS: Increasing each centered-view weight from 0.375 to 0.5 will retain at least 9,215 correct predictions and reduce validation cross-entropy below 0.222402 by better balancing unshifted evidence against translation-robust views.

INTENDED_EDIT: Increase both centered-view ensemble weights to 0.5 and update the normalization from 6.75 to 7.0, leaving training and all shifted-view weights unchanged.

EVIDENCE: Reducing centered weights to 0.25 preserved accuracy but worsened cross-entropy from 0.22289843 to 0.22299811; testing the equally sized upward perturbation is the informative opposite side of that bracket on the new 9,215-correct symmetric ensemble.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.375,
        )
=======
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.5
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.5,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
=======
        return (probability_sum / 7.0).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE