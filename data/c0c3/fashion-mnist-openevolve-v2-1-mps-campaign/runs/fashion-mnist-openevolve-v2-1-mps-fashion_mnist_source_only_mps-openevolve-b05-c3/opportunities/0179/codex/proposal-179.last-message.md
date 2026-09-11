MECHANISM: Qualified confidence-weighted TTA with finer post-ensemble calibration

HYPOTHESIS: Restoring 1.10× per-view sharpening will recover exactly 9,210 correct predictions, while a 1.1875× argmax-invariant final multiplier will reduce cross-entropy below 0.2228996.

INTENDED_EDIT: Restore the best verified per-view temperature and increase final ensemble sharpening halfway from the qualified 1.175× setting toward the unresolved 1.20× setting.

EVIDENCE: The 1.10× per-view, 1.175× final design achieved the best verified score with 9,210 correct and 0.2228996 cross-entropy; 1.125× per-view lost one correct prediction, while positive final scaling preserves argmax decisions.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.125, dim=1) * 0.375
=======
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
>>>>>>> REPLACE

<<<<<<< SEARCH
                self._forward_once(images.flip(-1)) * 1.125,
=======
                self._forward_once(images.flip(-1)) * 1.10,
>>>>>>> REPLACE

<<<<<<< SEARCH
            probability_sum.add_(F.softmax(view_logits * 1.125, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.125, dim=1))
=======
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.1875)
>>>>>>> REPLACE