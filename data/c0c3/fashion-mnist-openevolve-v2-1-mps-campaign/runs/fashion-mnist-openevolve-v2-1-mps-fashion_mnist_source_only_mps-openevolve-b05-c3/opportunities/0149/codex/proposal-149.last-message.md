MECHANISM: Fine-grained interior TTA weight interpolation

HYPOTHESIS: Reducing each unshifted-view weight from 0.5 to 0.375 will exceed 9,208 correct predictions by locating a better interior balance between the inferior zero-weight and equal-weight endpoints.

INTENDED_EDIT: Give the original and flipped unshifted predictions 0.375 weight each while retaining unit-weight shifted views, then normalize by total weight 8.75.

EVIDENCE: Unshifted weights 0.0, 0.5, and 1.0 produced 9,206, 9,208, and 9,206 correct respectively, demonstrating an interior optimum; the unresolved 0.25 attempts supplied no contrary performance result.

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
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.375,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 8.75).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE