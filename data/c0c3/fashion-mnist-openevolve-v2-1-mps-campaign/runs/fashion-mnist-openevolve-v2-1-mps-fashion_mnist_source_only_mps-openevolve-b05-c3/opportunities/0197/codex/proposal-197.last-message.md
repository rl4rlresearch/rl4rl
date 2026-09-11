MECHANISM: Fine-grained centered-view weight interpolation

HYPOTHESIS: Increasing each centered-view weight from 0.375 to 0.4 will preserve all 9,215 correct predictions while reducing validation cross-entropy below 0.222402.

INTENDED_EDIT: Raise both centered-view weights to 0.4 and adjust ensemble normalization from 6.75 to 6.8, leaving training, shifted-view weights, and calibration unchanged.

EVIDENCE: Centered weights of 0.25 and 0.5 each lost one correct prediction, while 0.5 reduced cross-entropy to 0.222354; a smaller upward step tests whether that calibration benefit begins before the accuracy boundary.

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
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.4
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.4,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
=======
        return (probability_sum / 6.8).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE