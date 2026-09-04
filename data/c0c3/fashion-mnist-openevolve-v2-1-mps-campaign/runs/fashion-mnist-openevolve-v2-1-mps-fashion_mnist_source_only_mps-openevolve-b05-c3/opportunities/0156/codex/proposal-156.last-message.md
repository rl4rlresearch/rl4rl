MECHANISM: Half-weight centered-view calibrated probability TTA

HYPOTHESIS: The qualified half-weight centered ensemble will reproduce 9,208 correct predictions, exceeding the current 9,206, while 1.10× final sharpening reduces cross-entropy.

INTENDED_EDIT: Give each centered view 0.5 weight, retain unit-weight shifted views, normalize by 9.0, and increase final sharpening from 1.05× to 1.10×.

EVIDENCE: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy; later verification failures supplied no contrary performance result.

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