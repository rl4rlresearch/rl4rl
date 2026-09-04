MECHANISM: Quarter-step reflected-orientation TTA calibration

HYPOTHESIS: Favoring reflected predictions by one quarter of the previously tested symmetric increment will preserve all 9,348 correct predictions while reducing validation cross-entropy below 0.18770656051635742.

INTENDED_EDIT: Keep the best crop weights and temperature, but lower the native-orientation coefficient to the immediate float32 predecessor of 1.0 while retaining unit reflected weight and exact coefficient-aware normalization.

EVIDENCE: Full reflected preference worsened cross-entropy by only 1.14e-9, whereas the equal-magnitude native preference worsened it by 3.43e-9; this asymmetric response estimates a shallow optimum about one quarter-step toward reflected predictions.

<<<<<<< SEARCH
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
=======
        native_weight = 0.999999940395355224609375
        reflected_weight = 1.0
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