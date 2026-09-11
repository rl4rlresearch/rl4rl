MECHANISM: Reliability-weighted geometric ensembling

HYPOTHESIS: Double-weighting the unshifted view while retaining the four useful axial translations will exceed 9,162 correct predictions because axial ensembling improved the centered ensemble, while adding less reliable diagonal views reduced accuracy.

INTENDED_EDIT: Weight the centered crop twice as strongly as each one-pixel axial crop during mirrored test-time logit averaging.

EVIDENCE: The centered-plus-axial ensemble achieved 9,162 correct versus 9,151 for centered-only evaluation, but the full grid fell to 9,159; this indicates that transformation views have unequal predictive reliability.

<<<<<<< SEARCH
        for crop in crops:
            logits = logits + self._forward_once(crop)
            logits = logits + self._forward_once(crop.flip(-1))
        return logits / (2.0 * len(crops))
=======
        crop_weights = (2.0, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            logits = logits + weight * self._forward_once(crop)
            logits = logits + weight * self._forward_once(crop.flip(-1))
        return logits / (2.0 * sum(crop_weights))
>>>>>>> REPLACE