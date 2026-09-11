MECHANISM: Continued confidence-weighted probability TTA

HYPOTHESIS: Restoring the qualified 1/3-centered training sampler and increasing per-view sharpening from 1.10× to 1.15× will exceed 9,210 correct predictions by further emphasizing views with decisive class margins, while retaining the best verified 1.15× final calibration.

INTENDED_EDIT: Use two centered outcomes and four cardinal shifts during terminal training, sharpen every centered and shifted TTA view by 1.15×, and apply the qualified 1.15× post-ensemble multiplier.

EVIDENCE: With the same 1/3-centered trajectory, increasing per-view sharpening from 1.05× to 1.10× retained 9,210 correct while reducing cross-entropy from 0.2237609 to 0.2230063; continuing this correctness-relevant adjustment is the most direct test beyond the current plateau.

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
        probability_sum = F.softmax(logits * 1.15, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.15,
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
            probability_sum.add_(F.softmax(view_logits * 1.15, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.15, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        directions = torch.randint(
            0, 6, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE