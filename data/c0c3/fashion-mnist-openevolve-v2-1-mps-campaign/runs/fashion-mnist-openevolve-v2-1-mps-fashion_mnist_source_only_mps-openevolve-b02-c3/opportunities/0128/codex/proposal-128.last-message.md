MECHANISM: One-ULP ensemble temperature sharpening

HYPOTHESIS: Restoring Reference Design 2’s best crop weights and lowering the positive temperature by one float32 ULP will preserve 9,348 correct predictions while reducing validation cross-entropy below 0.18770656127929689.

INTENDED_EDIT: Restore the best verified vertical and horizontal crop weights, then change the temperature from float32 0.753170013427734375 to its immediately lower representable value.

EVIDENCE: Reference Design 2 has the lowest verified cross-entropy, while the failed second horizontal ULP establishes its crop weighting as a discrete local optimum; positive temperature scaling preserves argmax predictions, making calibration the lowest-risk untested refinement.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.312042236328125,
            1.314910888671875,
            0.6865234375,
            0.6865234375,
        )
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
        ).log() / 0.75317
=======
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
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
        ).log() / 0.753169953823089599609375
>>>>>>> REPLACE