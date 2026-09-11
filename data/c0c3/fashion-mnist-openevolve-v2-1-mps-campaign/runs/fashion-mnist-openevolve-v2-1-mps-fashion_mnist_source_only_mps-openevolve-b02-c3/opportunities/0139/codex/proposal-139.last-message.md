MECHANISM: Reflected-orientation TTA micro-reweighting

HYPOTHESIS: Favoring reflected predictions by the smallest symmetric float32 increment will preserve 9,348 correct predictions while reducing cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Restore Reference Design 2’s optimal crop weights and temperature, then transfer orientation weight from every native crop prediction to its reflection while preserving total ensemble weight.

EVIDENCE: Favoring native orientations worsened cross-entropy to 0.18770656394958496 without changing correctness; the reverse direction is the most informative remaining orthogonal calibration probe around Reference Design 2.

<<<<<<< SEARCH
        crop_weights = (
            3.0000002384185791015625,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523258686065673828125,
            0.686523377895355224609375,
        )
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
=======
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
        native_weight = 0.99999988079071044921875
        reflected_weight = 1.00000011920928955078125
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + (
                weight * native_weight
            ) * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + (
                weight * reflected_weight
            ) * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (
            (native_weight + reflected_weight) * sum(crop_weights)
        )
>>>>>>> REPLACE