MECHANISM: Boundary-seeking centered-view interpolation

HYPOTHESIS: Centered-view weights of 0.3875 will preserve all 9,215 predictions achieved at 0.375 while slightly reducing cross-entropy toward the improvement observed at 0.5.

INTENDED_EDIT: Restore centered-view influence past the regressed 0.25 setting, using a smaller upward step than the unverified 0.4 experiment, and normalize the ensemble by its new total weight of 6.775.

EVIDENCE: Weight 0.375 produced the best result of 9,215 correct at 0.22240231 cross-entropy; 0.5 lost one prediction but improved cross-entropy to 0.22235397, while 0.25 also lost one and worsened cross-entropy. This motivates a conservative interpolation just above 0.375.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.25
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.25,
        )
=======
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.3875
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
                dim=1,
            ),
            alpha=0.3875,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.5).clamp_min(1e-8).log().mul_(1.178)
=======
        return (probability_sum / 6.775).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE