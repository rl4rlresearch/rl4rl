MECHANISM: Native-orientation TTA micro-reweighting

HYPOTHESIS: Restoring Reference Design 2’s best temperature and transferring the smallest symmetric float32 weight increment from reflected predictions to native-orientation predictions will retain all 9,348 correct predictions while reducing cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Restore the best verified temperature, then slightly favor each native crop over its horizontal reflection while preserving their exact combined orientation weight.

EVIDENCE: Reference Design 2 remains the lowest-cross-entropy implementation; temperature and crop-weight directions have been bracketed without changing correctness, while the currently equal native/reflected weighting is an untested orthogonal calibration dimension.

<<<<<<< SEARCH
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.75317013263702392578125
=======
        native_weight = 1.00000011920928955078125
        reflected_weight = 0.99999988079071044921875
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + native_weight * weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + reflected_weight * weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (
            (native_weight + reflected_weight) * sum(crop_weights)
        )
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE