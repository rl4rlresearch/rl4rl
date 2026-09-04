MECHANISM: Reduced centered-view influence in probability-space TTA

HYPOTHESIS: Lowering each centered-view weight from 0.375 to 0.25 will exceed 9,210 correct predictions by further favoring translation-robust shifted views while preserving the qualified training trajectory and per-view sharpening.

INTENDED_EDIT: Reduce only the original and horizontally flipped centered-view weights, updating the ensemble normalization from 6.75 to 6.50.

EVIDENCE: The weighted eight-view ensemble achieved 9,210 correct versus 9,206 for the equal-weight ten-view design, while final-scalar calibration cannot alter predictions; this isolates the still-untested centered-view weighting as the next accuracy-sensitive variable.

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
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.25
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.25,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
=======
        return (probability_sum / 6.50).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE