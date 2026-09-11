MECHANISM: Confidence-weighted TTA with stronger post-ensemble calibration

HYPOTHESIS: The qualified 1.10× per-view ensemble will retain exactly 9,210 correct predictions, while increasing its argmax-invariant final multiplier from 1.15× to 1.20× will reduce validation cross-entropy below 0.2230063.

INTENDED_EDIT: Restore the best verified 1.10× per-view sharpening and apply 1.20× final log-probability sharpening without changing training, parameters, or inference workload.

EVIDENCE: Reference Design 2 achieved 9,210 correct with 0.2230063 cross-entropy at 1.10× per-view and 1.15× final sharpening; earlier raising final sharpening from 1.10× to 1.15× preserved 9,210 correct and lowered cross-entropy, while the 1.20× attempts timed out without accuracy evidence against the change.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.375,
        )
=======
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.375,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.10)
=======
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.20)
>>>>>>> REPLACE