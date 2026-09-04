MECHANISM: Accuracy-boundary diagonal-view shrinkage

HYPOTHESIS: A diagonal weight of 0.7125 will retain the 9,192 correct predictions achieved at 0.725 while lowering cross-entropy toward the better-calibrated 0.70 ensemble.

INTENDED_EDIT: Weight the four diagonal translation views and their flips at 0.7125 while keeping centered and cardinal views at full weight, with normalization adjusted to 15.7.

EVIDENCE: Weight 0.725 achieved 9,192 correct at 0.237434 cross-entropy, while 0.70 achieved 9,191 correct at the lower 0.237421 cross-entropy; bisecting them directly probes the accuracy boundary for a tie-breaking calibration gain.

<<<<<<< SEARCH
        for view in shifted_views:
            logit_sum = logit_sum + self._forward_once(view)
            logit_sum = logit_sum + self._forward_once(view.flip(-1))
        return logit_sum / 18.0
=======
        diagonal_indices = {0, 2, 5, 7}
        for index, view in enumerate(shifted_views):
            weight = 0.7125 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.7
>>>>>>> REPLACE