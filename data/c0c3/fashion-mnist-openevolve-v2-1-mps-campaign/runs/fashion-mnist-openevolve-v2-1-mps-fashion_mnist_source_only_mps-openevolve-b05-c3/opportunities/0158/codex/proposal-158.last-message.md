MECHANISM: Shift-favored probability TTA

HYPOTHESIS: Half-weighting both centered views will reproduce the qualified 9,208-correct result, exceeding the current 9,206 correct predictions.

INTENDED_EDIT: Give each centered view 0.5 weight, retain unit weight for the eight shifted views, and normalize by total weight 9.0.

EVIDENCE: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy; subsequent verification failures reported no contrary accuracy result, while the zero- and unit-weight endpoints each achieved only 9,206 correct.

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
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE