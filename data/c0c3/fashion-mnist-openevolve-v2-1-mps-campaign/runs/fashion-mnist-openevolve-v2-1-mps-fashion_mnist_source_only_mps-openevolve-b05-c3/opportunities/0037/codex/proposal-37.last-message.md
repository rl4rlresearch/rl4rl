MECHANISM: Center-weighted test-time augmentation

HYPOTHESIS: Giving the two unshifted predictions twice the weight of shifted predictions will exceed 9,167 correct by preserving ensemble robustness while reducing influence from padded, spatially distorted views.

INTENDED_EDIT: Double the probability contribution of the original and horizontally flipped center views and renormalize the ten-view ensemble by twelve.

EVIDENCE: Restricting late training to center/cardinal one-pixel views improved correct predictions monotonically up to 9,167, indicating sensitivity to inference-view alignment; centered views most closely match the fixed validation framing.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )
=======
        probability_sum = 2.0 * F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1),
            alpha=2.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 12.0).clamp_min(1e-8).log()
>>>>>>> REPLACE