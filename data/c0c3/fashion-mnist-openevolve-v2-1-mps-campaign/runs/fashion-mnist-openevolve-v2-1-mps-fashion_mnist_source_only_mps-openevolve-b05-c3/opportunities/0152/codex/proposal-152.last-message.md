MECHANISM: TTA-matched terminal shift curriculum

HYPOTHESIS: Matching terminal augmentation frequencies to the best verified half-weight centered TTA will exceed 9,208 correct predictions by training on centered versus shifted views in the same 1:8 ratio used during validation.

INTENDED_EDIT: Restore the qualified half-weight centered ensemble with 1.10× calibration, and sample terminal crops with one centered outcome and two outcomes for each cardinal shift.

EVIDENCE: Reference Design 1 achieved 9,208 correct with total centered weight 1.0 and shifted weight 8.0, outperforming equal weighting and shift-only aggregation; the current uniform terminal sampler instead presents centered examples at a mismatched 1:4 ratio.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )
=======
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.5
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.5,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        direction_codes = torch.randint(
            0, 9, (images.shape[0],), device=images.device
        )
        directions = (direction_codes + 1) // 2
>>>>>>> REPLACE