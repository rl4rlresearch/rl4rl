MECHANISM: Clean-view-anchored probability TTA

HYPOTHESIS: Doubling the influence of the two unshifted views will exceed 9,206 correct predictions by preventing the eight padded shifts from dominating close ensemble decisions.

INTENDED_EDIT: Weight the original and horizontally flipped unshifted predictions twice, retain all shifted views and existing sharpening, and normalize the resulting twelve units of ensemble weight.

EVIDENCE: Unsharpened Reference Design 1 and the sharpened current design both produced exactly 9,206 correct predictions, so confidence scaling changed cross-entropy but not decisions; view weighting is an untested, decision-changing aggregation axis with no extra model evaluations.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )
=======
        probability_sum = F.softmax(logits * 1.05, dim=1).mul_(2.0)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=2.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 12.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE