MECHANISM: Further shift-view-favored probability TTA

HYPOTHESIS: Reducing each unshifted-view weight from 0.5 to 0.25 will exceed 9,208 correct predictions by continuing the observed improvement as that weight decreased from 2.0 to 1.0 to 0.5.

INTENDED_EDIT: Halve both unshifted-view weights to 0.25 and renormalize the ten-view probability ensemble by its total weight of 8.5.

EVIDENCE: Unshifted weights of 2.0, 1.0, and 0.5 yielded 9,202, 9,206, and 9,208 correct respectively; the previous 0.25 attempt timed out and provided no contrary performance evidence.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.5
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.5,
        )
=======
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.25
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.25,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 8.5).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE